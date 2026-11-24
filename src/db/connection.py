"""
connection.py: one place to open a database connection.
Reads DATABASE_URL from .env (never from code, never committed).
"""
import os
from pathlib import Path

import psycopg
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")


def get_conn(register_vectors=False):
    """register_vectors=True: numpy arrays can be passed as pgvector values (embeddings, vector search)."""
    url = os.environ.get("DATABASE_URL")
    if not url:
        raise RuntimeError("DATABASE_URL is not set: copy .env.example to .env and fill it in.")
    conn = psycopg.connect(url, connect_timeout=10)
    if register_vectors:
        from pgvector.psycopg import register_vector
        register_vector(conn)
    return conn
