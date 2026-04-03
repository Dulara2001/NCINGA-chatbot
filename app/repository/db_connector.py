#db_connector.py

"""
Database Connector

"""
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, scoped_session
from dotenv import load_dotenv
from contextlib import contextmanager
import logging

load_dotenv()
logger = logging.getLogger(__name__)

# Global engine and session factory (singleton pattern)
_engine = None
_Session = None


def get_engine():
    """
    Get or create the global PostgreSQL database engine.
    Reads POSTGRES_URL from environment.

    Expected format:
        POSTGRES_URL=postgresql+psycopg2://user:password@host:5432/dbname
    """
    global _engine
    if _engine is None:
        postgres_url = os.getenv("POSTGRES_URL", "")
        if not postgres_url:
            raise ValueError("POSTGRES_URL environment variable is not set")

        try:
            _engine = create_engine(
                postgres_url,
                pool_size=10,
                max_overflow=5,
                pool_timeout=30,
                pool_recycle=1800,
                pool_pre_ping=True,
                echo=False
            )
            logger.info("PostgreSQL engine created successfully")
        except Exception as e:
            logger.error(f"Failed to create PostgreSQL engine: {e}")
            raise

    return _engine


def get_session_factory():
    """
    Get or create the global scoped session factory.
    """
    global _Session
    if _Session is None:
        engine = get_engine()
        session_factory = sessionmaker(bind=engine)
        _Session = scoped_session(session_factory)
    return _Session


def get_session():
    """
    Get a new session instance.
    Returns: (engine, session)

    Usage in repository functions:
        engine, session = get_session()
        try:
            # do work
            session.commit()
        except Exception as e:
            session.rollback()
            raise
        finally:
            session.close()
    """
    engine = get_engine()
    Session = get_session_factory()
    return engine, Session()


@contextmanager
def get_db_session():
    """
    Context manager for database sessions with automatic cleanup.

    Usage:
        with get_db_session() as session:
            # do work
            # session.commit() is automatic on success
    """
    Session = get_session_factory()
    session = Session()
    try:
        yield session
        session.commit()
    except Exception as e:
        session.rollback()
        logger.error(f"Database session error: {e}")
        raise
    finally:
        session.close()