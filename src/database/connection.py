"""Database connection management."""

import os
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv

load_dotenv()


def get_database_url() -> str:
    # On Streamlit Community Cloud, credentials live in st.secrets["database"].
    # Locally, they come from .env.  Fall back to defaults for CI/testing.
    try:
        import streamlit as st
        db = st.secrets.get("database", {})
        if db:
            return (
                f"postgresql://{db.get('DB_USER', 'analyst')}:"
                f"{db.get('DB_PASSWORD', '')}@"
                f"{db.get('DB_HOST', 'localhost')}:"
                f"{db.get('DB_PORT', '5432')}/"
                f"{db.get('DB_NAME', 'bhavcopy_db')}"
                f"?sslmode=require&channel_binding=require"
            )
    except Exception:
        pass  # not running inside Streamlit — use env vars

    return (
        f"postgresql://{os.getenv('DB_USER', 'analyst').strip()}:"
        f"{os.getenv('DB_PASSWORD', '').strip()}@"
        f"{os.getenv('DB_HOST', 'localhost').strip()}:"
        f"{os.getenv('DB_PORT', '5432').strip()}/"
        f"{os.getenv('DB_NAME', 'bhavcopy_db').strip()}"
    )


_engine = None


def get_engine():
    global _engine
    if _engine is None:
        _engine = create_engine(
            get_database_url(),
            pool_size=5,
            max_overflow=10,
            pool_pre_ping=True,
        )
    return _engine


def get_session():
    engine = get_engine()
    Session = sessionmaker(bind=engine)
    return Session()


def test_connection() -> bool:
    try:
        with get_engine().connect() as conn:
            conn.execute(text("SELECT 1"))
        print("✅  Database connection successful")
        return True
    except Exception as e:
        print(f"❌  Database connection failed: {e}")
        return False


if __name__ == "__main__":
    test_connection()