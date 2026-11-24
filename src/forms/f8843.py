"""
f8843.py: a draft of IRS Form 8843 that the student checks, signs and mails themselves.

Three inputs, kept apart on purpose:
    facts     the interview's answers (visa type, more than 5 years, green card steps): these
              decide lines 12 and 13 and whether the form applies at all (same rules as the
              student module's escalations)
    details   what only the student knows and types into the form page (name, addresses,
              entry date, trips abroad, school): validated here, never sent to a model,
              never saved; the browser sends them again with every request
    spec      f8843_<revision>.json: where every field sits on the official PDF

build_plan() gives every field a status:
    filled      the value follows directly from what the student typed or answered
    confirm     computed or derived by us (days present, visa years, entry date, deadline):
                the student must tick it before the final PDF is made
    user_only   the student writes it by hand (passport number, SSN/ITIN, signature)
    missing     not known yet; the final PDF waits until it is

render_pdf(plan, "review") marks those statuses on the form, stamps it "REVIEW COPY - DO NOT
MAIL" and appends a checklist page. render_pdf(plan, "final", confirmed=...) is refused
(FormNotReady) until nothing is missing, no escalation applies, and every "confirm" item has
been ticked; it then returns pages 1-2 only, ready to print.

Nothing here files anything: the student signs and mails the form (Form 8843 cannot be e-filed).
"""
from __future__ import annotations

import json
import re
from datetime import date
from functools import lru_cache
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from xml.sax.saxutils import escape

from pypdf import PdfReader, PdfWriter
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.pdfgen import canvas
from reportlab.platypus import LongTable, Paragraph, SimpleDocTemplate, Spacer, TableStyle

from src.forms.presence import count_days_present
from src.workflow.engine import MODULES

FORMS = Path(__file__).resolve().parent
SPEC_FILE = FORMS / "f8843_2025.json"

FILLED, CONFIRM, USER_ONLY, MISSING = "filled", "confirm", "user_only", "missing"
FILING = "filing"   # the "how and when to mail" item, confirmed like a field

FONT, SIZE, MIN_SIZE = "Helvetica", 9.0, 6.0
VISA_LETTER = {"f_visa": "F", "j_visa": "J", "m_visa": "M"}
VISA_CLASS = re.compile(r"^[FJMQ]-1$")
MAX_TEXT = 200
MAX_TRIPS = 30

# Interview answers that stop the draft: (fact, value, rule id in the student module or None).
STOPS = [
    ("is_us_citizen", True, None, "U.S. citizens do not file Form 8843."),
    ("has_green_card", True, None, "Green card holders are resident aliens and do not file Form 8843."),
    ("student_visa_type", "q_visa", "esc_student_visa_q", None),
    ("student_visa_type", "other_visa", "esc_student_visa_other", None),
    ("visa_in_substantial_compliance", False, "esc_visa_not_in_compliance", None),
    ("exempt_student_more_than_5_years", True, "esc_exempt_more_than_5_years", None),
    ("applied_for_green_card", True, "esc_applied_for_green_card", None),
]

NOTES = {
    "tin": "Write your SSN or ITIN by hand if you have one; leave it blank if you have neither. "
           "This tool never asks for it.",
    "l3b": "Write your passport number by hand. This tool does not collect it.",
    "l1a": "Check the visa type and entry date against your I-94 record (i94.cbp.dhs.gov) or the "
           "entry stamp in your passport.",
    "l4a": "Computed from your entry date and trips abroad (the day you leave and the day you land "
           "both count). Check it against your I-94 travel history.",
    "l4a_manual": "The automatic day count is not available; count the days yourself.",
    "l4b": "As an exempt student for all of 2025, every day you were present in 2025 is excluded, "
           "so this equals line 4a for 2025.",
    "l4b_changed": "Your status changed in 2025; ask a preparer which 2025 days can be excluded.",
    "l10": "Ask your international student office whom to list here (usually the director of "
           "your academic program or your DSO).",
    "l11": "Filled from your first entry date and visa. If you were in the U.S. on an F, J, M or Q "
           "visa in an earlier stay, correct those years.",
    "l11_changed": "Your visa type changed: attach a statement with the new visa type and the date "
                   "you got it.",
    "with_return_address": "Leave blank: you mail Form 8843 together with your Form 1040-NR.",
    "with_return_signature": "No signature on Form 8843 when it is attached to Form 1040-NR.",
    "signature": "Sign and date by hand after you have checked every line.",
    "unsure": "You answered 'not sure' in the interview; a preparer must settle this first.",
    "not_asked": "The interview has not answered this yet; finish the student questions first.",
    "too_long": "This text does not fit the space on the form; shorten it.",
}


class FormDetailsError(ValueError):
    """The details the student typed are not usable (bad date, non-English text, ...)."""


class FormNotReady(RuntimeError):
    """The final PDF was asked for too early; reasons lists what is still open."""

    def __init__(self, reasons: list[str]):
        super().__init__("; ".join(reasons))
        self.reasons = reasons


@lru_cache(maxsize=1)
def load_spec() -> dict:
    return json.loads(SPEC_FILE.read_text(encoding="utf-8"))


def template_bytes(spec: dict | None = None) -> bytes:
    """The official blank form, checked against the hash in the spec (a new IRS revision fails loudly)."""
    spec = spec or load_spec()
    data = (FORMS / spec["template"]).read_bytes()
    if sha256(data).hexdigest() != spec["template_sha256"]:
        raise RuntimeError(f"{spec['template']} is not the form revision the field map was made for")
    return data


# ---------------------------------------------------------------- details from the student

def _text(raw: dict, key: str, limit: int = MAX_TEXT) -> str:
    value = raw.get(key)
    if value is None:
        return ""
    if not isinstance(value, str):
        raise FormDetailsError(f"{key} must be text")
    value = "\n".join(" ".join(line.split()) for line in value.strip().splitlines() if line.strip())
    if len(value) > limit:
        raise FormDetailsError(f"{key} is longer than {limit} characters")
    if any(not (32 <= ord(ch) < 127 or ch == "\n") for ch in value):
        raise FormDetailsError(f"{key}: use English letters, digits and punctuation (the IRS form is in English)")
    return value


def _date(raw: dict, key: str) -> date | None:
    value = raw.get(key)
    if value in (None, ""):
        return None
    try:
        return date.fromisoformat(str(value))
    except ValueError:
        raise FormDetailsError(f"{key} must be a date like 2024-08-15") from None


def _flag(raw: dict, key: str) -> bool | None:
    value = raw.get(key)
    if value is None or isinstance(value, bool):
        return value
    raise FormDetailsError(f"{key} must be true, false or empty")


def clean_details(raw: dict | None) -> dict:
    """Validate what the student typed; returns plain values (dates as date objects)."""
    raw = raw or {}
    details = {key: _text(raw, key) for key in
               ("first_name", "last_name", "address_abroad", "address_us", "citizenship",
                "passport_country", "school", "program_director", "previous_status")}
    for key in ("first_name", "last_name", "citizenship", "passport_country", "previous_status"):
        if "\n" in details[key]:
            raise FormDetailsError(f"{key} must be one line")
    visa_class = _text(raw, "visa_class", 10).upper()
    if visa_class and not VISA_CLASS.match(visa_class):
        raise FormDetailsError("visa_class must be F-1, J-1, M-1 or Q-1")
    details["visa_class"] = visa_class
    details["first_entry"] = _date(raw, "first_entry")
    details["status_change_date"] = _date(raw, "status_change_date")
    details["status_changed"] = bool(_flag(raw, "status_changed"))
    details["has_us_income"] = _flag(raw, "has_us_income")

    trips = raw.get("trips") or []
    if not isinstance(trips, list) or len(trips) > MAX_TRIPS:
        raise FormDetailsError(f"trips must be a list of at most {MAX_TRIPS} trips")
    pairs = []
    for i, trip in enumerate(trips, 1):
        if not isinstance(trip, dict):
            raise FormDetailsError(f"trip {i} must have left and returned dates")
        left, back = _date(trip, "left"), _date(trip, "returned")
        if left is None or back is None or back <= left:
            raise FormDetailsError(f"trip {i}: the return date must be after the departure date")
        pairs.append((left, back))
    pairs.sort()
    for (_, back), (left, _) in zip(pairs, pairs[1:]):
        if left < back:
            raise FormDetailsError("two trips overlap; check the dates")
    if pairs and details["first_entry"] and pairs[0][0] < details["first_entry"]:
        raise FormDetailsError("a trip starts before your first entry date")
    details["trips"] = pairs
    return details


# ---------------------------------------------------------------- text fitting (plan and render agree)

def _wrap(text: str, width: float, size: float) -> list[str]:
    lines = []
    for paragraph in text.split("\n"):
        line = ""
        for word in paragraph.split(" "):
            trial = f"{line} {word}".strip()
            if line and stringWidth(trial, FONT, size) > width:
                lines.append(line)
                line = word
            else:
                line = trial
        lines.append(line)
    return lines


def fit(text: str, field: dict) -> tuple[list[str], float, bool]:
    """Lines and font size that fit the field; ok is False when even the smallest size overflows."""
    rows = len(field.get("tops", [None]))
    size = SIZE
    while True:
        lines = _wrap(text, field["width"], size) if rows > 1 else [text.replace("\n", ", ")]
        if len(lines) <= rows and all(stringWidth(line, FONT, size) <= field["width"] for line in lines):
            return lines, size, True
        if size <= MIN_SIZE:
            return lines[:rows], size, False
        size -= 0.5


# ---------------------------------------------------------------- the plan

def confirm_key(field_id: str) -> str:
    """Fields the student confirms together: the three line 4a years are one item, the line 11 years another."""
    return re.sub(r"_\d{4}$", "", field_id)


def _escalation_reasons() -> dict:
    module = json.loads((MODULES / "student" / "module.json").read_text(encoding="utf-8"))
    return {rule["id"]: rule["reason"] for rule in module["escalation_rules"]}


def _us_date(value: date) -> str:
    return value.strftime("%m/%d/%Y")


def build_plan(facts: dict, details: dict | None = None, unsure: list | None = None) -> dict:
    """Every field of the form with its value, status and a note for the student.

    facts    interview facts (State.facts); details: clean_details() output or None (MCP: the
             assistant never collects names or addresses, so those fields come back "missing").
    Returns {"form", "revision", "applicable", "stops", "filing", "fields", "to_confirm",
             "user_only", "missing", "ready_for_final"}.
    """
    spec = load_spec()
    d = details if details is not None else clean_details({})
    unsure = set(unsure or [])
    reasons = _escalation_reasons()
    stops = [{"rule": rule, "reason": text or reasons.get(rule, rule)}
             for fact, value, rule, text in STOPS if facts.get(fact) == value]
    applicable = not any(fact in ("is_us_citizen", "has_green_card") and facts.get(fact) is True
                         for fact, *_ in STOPS)

    values: dict[str, tuple] = {}

    def put(field_id, value, status, note=""):
        values[field_id] = (value, status, note)

    def typed(field_id, value, note_if_missing="Fill this in on the form page."):
        put(field_id, value, FILLED if value else MISSING, "" if value else note_if_missing)

    def answered(field_id, fact):
        value = facts.get(fact)
        if value is None:
            put(field_id, "", MISSING, NOTES["unsure"] if fact in unsure else NOTES["not_asked"])
        else:
            put(field_id, "Yes" if value else "No", FILLED)

    typed("first_name", d["first_name"])
    typed("last_name", d["last_name"])
    put("tin", "", USER_ONLY, NOTES["tin"])

    alone = None if d["has_us_income"] is None else not d["has_us_income"]
    for key in ("address_abroad", "address_us"):
        if alone is None:
            put(key, "", MISSING, "Say whether you had U.S. income in 2025 first.")
        elif alone:
            typed(key, d[key])
        else:
            put(key, "", FILLED, NOTES["with_return_address"])

    visa = d["visa_class"]
    entry = d["first_entry"]
    interview_letter = VISA_LETTER.get(facts.get("student_visa_type"))
    if visa and entry:
        note = NOTES["l1a"]
        if interview_letter and visa[0] != interview_letter:
            note = (f"Your interview answer was an {interview_letter} visa but you typed {visa}; "
                    "fix whichever is wrong. ") + note
        put("l1a", f"{visa}  {_us_date(entry)}", CONFIRM, note)
    else:
        put("l1a", "", MISSING, "Enter your visa type and the date you entered the United States.")

    changed = d["status_changed"]
    if not visa:
        put("l1b", "", MISSING, "Enter your visa type.")
    elif changed:
        if d["previous_status"] and d["status_change_date"]:
            put("l1b", f"{visa} (changed from {d['previous_status']} on {_us_date(d['status_change_date'])})",
                CONFIRM, "Check the previous status and the date of change.")
        else:
            put("l1b", "", MISSING, "Enter your previous status and the date it changed.")
    else:
        put("l1b", visa, FILLED)

    typed("l2", d["citizenship"])
    typed("l3a", d["passport_country"])
    put("l3b", "", USER_ONLY, NOTES["l3b"])

    days_2025 = None
    for year in spec["years"]["4a"]:
        field_id = f"l4a_{year}"
        if entry is None:
            put(field_id, "", MISSING, "Enter the date you entered the United States.")
            continue
        try:
            days = count_days_present(entry, d["trips"], year)
        except NotImplementedError:
            put(field_id, "", USER_ONLY, NOTES["l4a_manual"])
            continue
        put(field_id, str(days), CONFIRM, NOTES["l4a"])
        if year == spec["tax_year"]:
            days_2025 = days
    if changed:
        put("l4b", "", USER_ONLY, NOTES["l4b_changed"])
    elif days_2025 is None:
        status = MISSING if entry is None else USER_ONLY
        put("l4b", "", status, values[f"l4a_{spec['tax_year']}"][2])
    else:
        put("l4b", str(days_2025), CONFIRM, NOTES["l4b"])

    typed("l9", d["school"], "Enter your school's name, address and telephone number.")
    if d["program_director"]:
        put("l10", d["program_director"], CONFIRM, NOTES["l10"])
    else:
        put("l10", "", USER_ONLY, NOTES["l10"])

    for year in spec["years"]["11"]:
        field_id = f"l11_{year}"
        if entry is None or not visa:
            put(field_id, "", MISSING, "Enter your visa type and entry date.")
            continue
        held = visa
        if changed and d["status_change_date"] and year < d["status_change_date"].year:
            held = d["previous_status"] if VISA_CLASS.match(d["previous_status"].upper()) else ""
        value = held if year >= entry.year else ""
        put(field_id, value, CONFIRM, NOTES["l11_changed"] if changed else NOTES["l11"])

    answered("l12", "exempt_student_more_than_5_years")
    answered("l13", "applied_for_green_card")
    put("l14", "", FILLED)
    for key in ("signature", "signature_date"):
        if alone is False:
            put(key, "", FILLED, NOTES["with_return_signature"])
        else:
            put(key, "", USER_ONLY, NOTES["signature"])

    fields = []
    for field in spec["fields"]:
        value, status, note = values[field["id"]]
        if value and field.get("kind") != "checkbox" and not fit(value, field)[2]:
            status, note = MISSING, NOTES["too_long"]
        fields.append({"id": field["id"], "line": field["line"], "label": field["label"],
                       "value": value, "status": status, "note": note})

    due = spec["filing"]["alone_due" if alone else "with_return_due"]
    if alone is None:
        filing = {"status": MISSING, "note": "Say whether you had U.S. income in 2025 (wages, taxable scholarship, ...)."}
    elif alone:
        filing = {"status": CONFIRM, "alone": True, "due": due, "mail_to": spec["filing"]["mail_to"],
                  "note": f"Mail pages 1-2, signed and dated, by itself to the address below by {due}. "
                          "Form 8843 cannot be e-filed. Keep a copy."}
    else:
        filing = {"status": CONFIRM, "alone": False, "due": due,
                  "note": f"Attach Form 8843 to your Form 1040-NR, due {due}. A nonresident return needs a "
                          "preparer or software that supports Form 1040-NR; this tool does not prepare it."}

    to_confirm = list(dict.fromkeys(confirm_key(f["id"]) for f in fields if f["status"] == CONFIRM))
    if filing["status"] == CONFIRM:
        to_confirm.append(FILING)
    missing = [f["id"] for f in fields if f["status"] == MISSING]
    if filing["status"] == MISSING:
        missing.append(FILING)
    return {
        "form": spec["form"], "revision": spec["revision"], "tax_year": spec["tax_year"],
        "applicable": applicable, "stops": stops, "filing": filing, "fields": fields,
        "to_confirm": to_confirm,
        "user_only": [f["id"] for f in fields if f["status"] == USER_ONLY],
        "missing": missing,
        "ready_for_final": applicable and not stops and not missing,
    }


def final_blockers(plan: dict, confirmed: set[str] | list[str]) -> list[str]:
    """Why the final PDF cannot be made yet (empty list: it can)."""
    reasons = []
    if not plan["applicable"]:
        reasons.append("Form 8843 does not apply: " + "; ".join(s["reason"] for s in plan["stops"]))
        return reasons
    reasons += [f"Needs a preparer: {s['reason']}" for s in plan["stops"]]
    labels = {f["id"]: f"line {f['line']} ({f['label']})" for f in plan["fields"]} | {FILING: "how and when to mail"}
    labels |= {confirm_key(f["id"]): f"line {f['line']}" for f in plan["fields"] if confirm_key(f["id"]) != f["id"]}
    open_items = [labels[i] for i in plan["missing"]]
    if open_items:
        reasons.append("Still missing: " + "; ".join(open_items))
    unconfirmed = [labels[i] for i in plan["to_confirm"] if i not in set(confirmed)]
    if unconfirmed:
        reasons.append("Not confirmed yet: " + "; ".join(unconfirmed))
    return reasons


# ---------------------------------------------------------------- PDF

HIGHLIGHT = {CONFIRM: colors.HexColor("#FFE066"), USER_ONLY: colors.HexColor("#9DB4C0"),
             MISSING: colors.HexColor("#F08080")}
STATUS_WORDS = {FILLED: "Filled", CONFIRM: "Please confirm", USER_ONLY: "You write this by hand",
                MISSING: "Missing"}


def _field_box(field: dict, height: float) -> tuple[float, float, float, float]:
    """(x, y, w, h) in PDF points covering the field's writing area."""
    if field.get("kind") == "checkbox":
        xs = [box[0] for box in field["boxes"].values()]
        top = min(box[1] for box in field["boxes"].values())
        return min(xs) - 2, height - top - field["size"] - 2, max(xs) - min(xs) + field["size"] + 22, field["size"] + 4
    tops = field.get("tops") or [field["top"]]
    return field["x"] - 2, height - tops[-1] - 1, field["width"] + 4, tops[-1] - tops[0] + 11


def _draw_value(c: canvas.Canvas, field: dict, value: str, height: float) -> None:
    if field.get("kind") == "checkbox":
        if value not in field["boxes"]:
            return
        x, top = field["boxes"][value]
        size, y = field["size"], height - top - field["size"]
        c.setLineWidth(1.1)
        c.line(x + 1.6, y + 1.6, x + size - 1.6, y + size - 1.6)
        c.line(x + 1.6, y + size - 1.6, x + size - 1.6, y + 1.6)
        return
    lines, size, _ = fit(value, field)
    c.setFont(FONT, size)
    for line, top in zip(lines, field.get("tops") or [field["top"]]):
        c.drawString(field["x"], height - top + 2.2, line)


def _overlay(spec: dict, page_index: int, plan: dict, mode: str) -> bytes:
    height = spec["page_height"]
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=(612, height))
    by_id = {f["id"]: f for f in plan["fields"]}
    numbers = {group[0]["id"]: n for n, (_, group) in enumerate(_review_groups(plan), 1)}
    for field in spec["fields"]:
        if field["page"] != page_index:
            continue
        item = by_id[field["id"]]
        if mode == "review" and item["status"] in HIGHLIGHT:
            x, y, w, h = _field_box(field, height)
            c.saveState()
            c.setFillColor(HIGHLIGHT[item["status"]])
            c.setFillAlpha(0.45)
            c.rect(x, y, w, h, stroke=0, fill=1)
            c.restoreState()
            if field["id"] in numbers:
                c.saveState()
                c.setFont("Helvetica-Bold", 6.5)
                c.setFillColor(colors.HexColor("#B42318"))
                c.drawString(x + w + 1.5, y + h - 6, str(numbers[field["id"]]))
                c.restoreState()
        if item["value"] and (mode == "review" or item["status"] in (FILLED, CONFIRM)):
            c.setFillColor(colors.HexColor("#0B2A6F") if mode == "review" else colors.black)
            _draw_value(c, field, item["value"], height)
    if mode == "review":
        c.saveState()
        c.setFillColor(colors.HexColor("#B42318"))
        c.setFillAlpha(0.13)
        c.setFont("Helvetica-Bold", 46)
        c.translate(306, height / 2)
        c.rotate(35)
        c.drawCentredString(0, 0, "REVIEW COPY - DO NOT MAIL")
        c.restoreState()
    c.showPage()
    c.save()
    return buffer.getvalue()


def _review_groups(plan: dict) -> list[tuple[str, list[dict]]]:
    """What the student has to act on, in form order: one group per line (the line 4a years together)."""
    groups: dict[str, list[dict]] = {}
    for field in plan["fields"]:
        if field["status"] in (CONFIRM, USER_ONLY, MISSING):
            key = "signature" if field["id"] == "signature_date" else confirm_key(field["id"])
            groups.setdefault(key, []).append(field)
    return list(groups.items())


def _group_value(fields: list[dict]) -> str:
    if len(fields) == 1:
        return fields[0]["value"] or "-"
    if all(re.search(r"_\d{4}$", f["id"]) for f in fields):
        return " | ".join(f"{f['id'][-4:]}: {f['value'] or 'blank'}" for f in fields)
    return " | ".join(f["value"] for f in fields if f["value"]) or "-"


def _checklist(plan: dict) -> bytes:
    body = ParagraphStyle("Body", fontName=FONT, fontSize=8.5, leading=11.5)
    bold = ParagraphStyle("Bold", parent=body, fontName="Helvetica-Bold")
    title = ParagraphStyle("Title", parent=body, fontName="Helvetica-Bold", fontSize=15, leading=19, spaceAfter=6)
    p = lambda text, style=body: Paragraph(escape(text), style)  # noqa: E731
    story = [p("Form 8843 draft: what you still have to check", title),
             p("This page and the marked form are a review copy. Do not mail them. Tick every yellow item on the "
               "form page, then download the final PDF, check it once more, sign it and mail pages 1-2. "
               "Nothing has been filed for you and this is not tax advice."),
             Spacer(1, 8)]
    if plan["stops"]:
        story += [p("Stop: talk to a preparer before you mail this form", bold)]
        story += [p("- " + s["reason"]) for s in plan["stops"]] + [Spacer(1, 6)]
    rows = [[p("#", bold), p("Line", bold), p("On the draft", bold), p("Status", bold), p("What to do", bold)]]
    for n, (_, group) in enumerate(_review_groups(plan), 1):
        first = group[0]
        rows.append([p(str(n)), p(first["line"]), p(_group_value(group)), p(STATUS_WORDS[first["status"]]),
                     p(first["note"])])
    filing = plan["filing"]
    rows.append([p("-"), p("Mailing"), p(filing.get("due", "-")), p(STATUS_WORDS[filing["status"]]),
                 p(filing["note"] + (" " + filing["mail_to"] if filing.get("mail_to") else ""))])
    table = LongTable(rows, colWidths=[22, 46, 128, 70, 254], repeatRows=1)
    table.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#C9D3D8")),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#EEF3F5")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    story.append(table)
    buffer = BytesIO()
    SimpleDocTemplate(buffer, pagesize=letter, leftMargin=46, rightMargin=46, topMargin=46, bottomMargin=46,
                      title="Form 8843 review checklist").build(story)
    return buffer.getvalue()


def render_pdf(plan: dict, mode: str = "review", confirmed: set[str] | list[str] | None = None) -> bytes:
    """The review copy (marked, with checklist) or the final copy (pages 1-2, only when ready)."""
    if mode not in ("review", "final"):
        raise ValueError("mode must be review or final")
    if mode == "final":
        reasons = final_blockers(plan, confirmed or [])
        if reasons:
            raise FormNotReady(reasons)
    spec = load_spec()
    template = PdfReader(BytesIO(template_bytes(spec)))
    writer = PdfWriter()
    for index in spec["mail_pages"]:
        page = writer.add_page(template.pages[index])
        page.merge_page(PdfReader(BytesIO(_overlay(spec, index, plan, mode))).pages[0])
    if mode == "review":
        for page in PdfReader(BytesIO(_checklist(plan))).pages:
            writer.add_page(page)
    writer.add_metadata({"/Title": f"Form 8843 ({spec['revision']}) - {'review copy' if mode == 'review' else 'final'}"})
    out = BytesIO()
    writer.write(out)
    return out.getvalue()
