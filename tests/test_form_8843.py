"""Form 8843 draft: day counting, the field plan, the confirmation gate and the PDFs (fictional students only)."""
from datetime import date, timedelta
from io import BytesIO
import unittest
from unittest.mock import patch

from pypdf import PdfReader

from src.forms import f8843
from src.forms.presence import count_days_present

F1_FACTS = {"is_us_citizen": False, "has_green_card": False, "student_visa_type": "f_visa",
            "visa_in_substantial_compliance": True, "exempt_student_more_than_5_years": False,
            "applied_for_green_card": False}
DETAILS = {
    "first_name": "Lin", "last_name": "Testcase", "visa_class": "F-1", "first_entry": "2024-08-15",
    "trips": [{"left": "2024-12-20", "returned": "2025-01-10"}],
    "citizenship": "Exampleland", "passport_country": "Exampleland",
    "address_abroad": "1 Sample Road\nSample City", "address_us": "100 Test Ave\nLos Angeles, CA 90007",
    "school": "Example University, 1 Campus Dr, Los Angeles, CA 90089, (213) 555-0100",
    "program_director": "Dr. A. Example, EE Program, 1 Campus Dr, Los Angeles, CA 90089, (213) 555-0101",
    "has_us_income": False,
}


def simple_count(first_entry, trips, year, until=None):
    """Reference count used by the plan tests, so they do not depend on presence.py."""
    start, end = max(first_entry, date(year, 1, 1)), min(until or date(year, 12, 31), date(year, 12, 31))
    if start > end:
        return 0
    days = (end - start).days + 1
    for left, back in trips:
        a, b = max(left + timedelta(1), start), min(back - timedelta(1), end)
        if a <= b:
            days -= (b - a).days + 1
    return days


def text_of(pdf: bytes) -> str:
    return "\n".join(page.extract_text() for page in PdfReader(BytesIO(pdf)).pages)


class CountDaysPresentTests(unittest.TestCase):
    """presence.count_days_present: the day count behind line 4a (implement it to make these pass)."""

    def test_whole_year_without_trips(self):
        self.assertEqual(count_days_present(date(2023, 9, 1), [], 2025), 365)

    def test_counts_from_first_entry_in_the_year(self):
        self.assertEqual(count_days_present(date(2024, 8, 15), [], 2024), 139)

    def test_not_yet_in_the_us(self):
        self.assertEqual(count_days_present(date(2024, 8, 15), [], 2023), 0)

    def test_departure_and_return_days_count(self):
        trip = [(date(2025, 6, 1), date(2025, 6, 20))]          # June 2-19 abroad: 18 days
        self.assertEqual(count_days_present(date(2023, 9, 1), trip, 2025), 365 - 18)

    def test_trip_across_new_year(self):
        trip = [(date(2024, 12, 20), date(2025, 1, 10))]
        self.assertEqual(count_days_present(date(2024, 8, 15), trip, 2025), 365 - 9)   # Jan 1-9
        self.assertEqual(count_days_present(date(2024, 8, 15), trip, 2024), 139 - 11)  # Dec 21-31

    def test_left_for_good(self):
        self.assertEqual(count_days_present(date(2023, 9, 1), [], 2025, until=date(2025, 5, 31)), 151)


class StudentPlaybookTests(unittest.TestCase):
    """playbooks/student_8843: the Form 8843 questions on their own, chosen on the interview page."""

    def test_questions_match_the_modules_word_for_word(self):
        from src.workflow.engine import load_scenario
        from src.workflow.guide_evidence import fingerprint
        _, own = load_scenario("student_8843")
        _, intake = load_scenario("intake")
        prints = {fingerprint(q) for q in intake}
        self.assertTrue(all(fingerprint(q) in prints for q in own))   # prebuilt passages still apply

    def test_f1_run_asks_only_the_form_questions_and_fills_the_plan(self):
        from src.workflow.engine import start_interview, step
        stub = lambda q: {"matches": []}
        state = start_interview("student_8843")
        state, out = step(state, search=stub)
        asked = []
        answers = {"is_us_citizen": "no", "has_green_card": "no", "student_visa_type": "F visa",
                   "visa_in_substantial_compliance": "yes", "exempt_student_more_than_5_years": "no",
                   "applied_for_green_card": "no"}
        while out["type"] == "question":
            asked.append(out["fact"])
            state, out = step(state, answers[out["fact"]], search=stub)
        self.assertEqual(asked, list(answers))
        self.assertEqual(state.facts, F1_FACTS)
        plan = f8843.build_plan(state.facts, None, state.unsure)
        self.assertTrue(plan["applicable"])
        self.assertEqual(plan["stops"], [])

    def test_citizen_ends_after_one_question(self):
        from src.workflow.engine import start_interview, step
        stub = lambda q: {"matches": []}
        state, out = step(start_interview("student_8843"), search=stub)
        state, out = step(state, "yes", search=stub)
        self.assertNotEqual(out["type"], "question")
        self.assertFalse(f8843.build_plan(state.facts)["applicable"])


@patch.object(f8843, "count_days_present", simple_count)
class PlanTests(unittest.TestCase):
    def plan(self, facts=None, **changes):
        return f8843.build_plan(facts or F1_FACTS, f8843.clean_details(DETAILS | changes))

    def field(self, plan, field_id):
        return next(f for f in plan["fields"] if f["id"] == field_id)

    def test_every_spec_field_is_planned(self):
        plan = self.plan()
        self.assertEqual([f["id"] for f in plan["fields"]], [f["id"] for f in f8843.load_spec()["fields"]])

    def test_f1_student_without_income(self):
        plan = self.plan()
        self.assertTrue(plan["applicable"])
        self.assertEqual(plan["stops"], [])
        self.assertEqual(plan["missing"], [])
        self.assertTrue(plan["ready_for_final"])
        self.assertEqual(self.field(plan, "l1a")["value"], "F-1  08/15/2024")
        self.assertEqual([self.field(plan, f"l4a_{y}")["value"] for y in (2025, 2024, 2023)], ["356", "128", "0"])
        self.assertEqual(self.field(plan, "l4b")["value"], "356")
        self.assertEqual(self.field(plan, "l11_2024")["value"], "F-1")
        self.assertEqual(self.field(plan, "l11_2023")["value"], "")
        self.assertEqual(self.field(plan, "l12")["value"], "No")
        self.assertEqual(plan["to_confirm"], ["l1a", "l4a", "l4b", "l10", "l11", "filing"])
        self.assertEqual(plan["user_only"], ["tin", "l3b", "signature", "signature_date"])
        self.assertEqual(plan["filing"]["due"], "2026-06-15")

    def test_with_income_the_form_goes_with_1040nr(self):
        plan = self.plan(has_us_income=True)
        self.assertEqual(self.field(plan, "address_us")["value"], "")
        self.assertEqual(self.field(plan, "signature")["status"], f8843.FILLED)
        self.assertEqual(plan["filing"]["due"], "2026-04-15")
        self.assertFalse(plan["filing"]["alone"])

    def test_green_card_holder_does_not_file(self):
        plan = self.plan(F1_FACTS | {"has_green_card": True})
        self.assertFalse(plan["applicable"])
        self.assertFalse(plan["ready_for_final"])

    def test_more_than_five_years_needs_a_preparer(self):
        plan = self.plan(F1_FACTS | {"exempt_student_more_than_5_years": True})
        self.assertTrue(plan["applicable"])
        self.assertEqual([s["rule"] for s in plan["stops"]], ["esc_exempt_more_than_5_years"])
        self.assertEqual(self.field(plan, "l12")["value"], "Yes")
        self.assertFalse(plan["ready_for_final"])

    def test_unsure_answer_stays_open(self):
        facts = dict(F1_FACTS)
        del facts["applied_for_green_card"]
        plan = f8843.build_plan(facts, f8843.clean_details(DETAILS), ["applied_for_green_card"])
        self.assertEqual(self.field(plan, "l13")["status"], f8843.MISSING)
        self.assertIn("not sure", self.field(plan, "l13")["note"])
        self.assertIn("l13", plan["missing"])

    def test_visa_that_contradicts_the_interview_is_flagged(self):
        note = self.field(self.plan(visa_class="J-1"), "l1a")["note"]
        self.assertIn("F visa", note)

    def test_status_change_fills_earlier_years_with_previous_visa(self):
        plan = self.plan(first_entry="2023-08-20", trips=[], status_changed=True,
                         previous_status="J-1", status_change_date="2025-02-01")
        self.assertEqual(self.field(plan, "l11_2023")["value"], "J-1")
        self.assertEqual(self.field(plan, "l4b")["status"], f8843.USER_ONLY)

    def test_details_are_validated(self):
        for bad in ({"first_entry": "15/08/2024"}, {"first_name": "林"}, {"visa_class": "H-1B"},
                    {"trips": [{"left": "2025-01-10", "returned": "2025-01-01"}]},
                    {"trips": [{"left": "2025-01-01", "returned": "2025-02-01"},
                               {"left": "2025-01-15", "returned": "2025-03-01"}]},
                    {"first_entry": "2025-01-01", "trips": [{"left": "2024-12-01", "returned": "2024-12-20"}]}):
            with self.subTest(bad=bad), self.assertRaises(f8843.FormDetailsError):
                f8843.clean_details(DETAILS | bad)

    def test_text_too_long_for_the_box_is_missing(self):
        plan = self.plan(school=("WWWWWWWWWW " * 18).strip())
        self.assertEqual(self.field(plan, "l9")["status"], f8843.MISSING)


class DayCountNotAvailableTests(unittest.TestCase):
    def test_days_become_handwritten_when_the_count_is_unavailable(self):
        with patch.object(f8843, "count_days_present", side_effect=NotImplementedError):
            plan = f8843.build_plan(F1_FACTS, f8843.clean_details(DETAILS))
        statuses = {f["id"]: f["status"] for f in plan["fields"]}
        self.assertEqual(statuses["l4a_2025"], f8843.USER_ONLY)
        self.assertEqual(statuses["l4b"], f8843.USER_ONLY)
        self.assertTrue(plan["ready_for_final"])


@patch.object(f8843, "count_days_present", simple_count)
class PdfTests(unittest.TestCase):
    def setUp(self):
        self.plan = f8843.build_plan(F1_FACTS, f8843.clean_details(DETAILS))

    def test_review_copy_is_marked_and_has_a_checklist(self):
        pdf = f8843.render_pdf(self.plan, "review")
        self.assertEqual(len(PdfReader(BytesIO(pdf)).pages), 3)
        text = text_of(pdf)
        self.assertIn("REVIEW COPY", text)
        self.assertIn("what you still have to check", text)

    def test_final_waits_for_every_confirmation(self):
        with self.assertRaises(f8843.FormNotReady) as raised:
            f8843.render_pdf(self.plan, "final", confirmed=["l1a"])
        self.assertIn("Not confirmed yet", raised.exception.reasons[0])
        self.assertNotIn("line 1a", raised.exception.reasons[0])

    def test_final_has_the_two_mailed_pages_without_marks(self):
        pdf = f8843.render_pdf(self.plan, "final", confirmed=self.plan["to_confirm"])
        self.assertEqual(len(PdfReader(BytesIO(pdf)).pages), 2)
        text = text_of(pdf)
        self.assertIn("Testcase", text)
        self.assertIn("08/15/2024", text)
        self.assertNotIn("REVIEW COPY", text)

    def test_final_refused_when_a_preparer_is_needed(self):
        plan = f8843.build_plan(F1_FACTS | {"applied_for_green_card": True}, f8843.clean_details(DETAILS))
        with self.assertRaises(f8843.FormNotReady) as raised:
            f8843.render_pdf(plan, "final", confirmed=plan["to_confirm"])
        self.assertIn("Needs a preparer", raised.exception.reasons[0])


class TemplateTests(unittest.TestCase):
    """The field map fits the official PDF it was made for (a new IRS revision must fail here)."""

    def test_template_hash_matches_the_spec(self):
        f8843.template_bytes()

    def test_anchor_text_is_where_the_map_expects_it(self):
        try:
            import pdfplumber
        except ImportError:
            self.skipTest("pdfplumber not installed")
        spec = f8843.load_spec()
        with pdfplumber.open(f8843.FORMS / spec["template"]) as pdf:
            for anchor in spec["anchors"]:
                words = pdf.pages[anchor["page"]].extract_words()
                with self.subTest(anchor=anchor["text"]):
                    self.assertTrue(any(w["text"] == anchor["text"] and abs(w["x0"] - anchor["x0"]) < 2
                                        and abs(w["top"] - anchor["top"]) < 2 for w in words))


if __name__ == "__main__":
    unittest.main()
