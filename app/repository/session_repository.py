#session_repository.py

"""
Session Repository
Handles database operations for storing and retrieving session records.

"""
import os
import json
import logging
from typing import Optional, List, Dict, Any
from datetime import datetime
from sqlalchemy import Column, Integer, String, Text, DateTime, Float, func, Index
from dotenv import load_dotenv
from app.repository.db_connector import get_session, get_engine
from app.schemas.base import Base

load_dotenv()

logger = logging.getLogger(__name__)

SESSION_TABLE_NAME = os.getenv("SESSION_TABLE_NAME", "sessions")

_model_cache: Dict[str, Any] = {}


class SessionRepository:
    """Repository for managing session records in PostgreSQL."""

    def __init__(self, table_name: Optional[str] = None):
        self.table_name = table_name or SESSION_TABLE_NAME

        if not self.table_name:
            raise ValueError(
                "SESSION_TABLE_NAME must be set in environment variables or provided"
            )

        self.Model = self._create_model()

    def _create_model(self):
        """
        Dynamically create a SQLAlchemy model for the sessions table.
        Cached to avoid redefinition across requests.
        """
        if self.table_name in _model_cache:
            return _model_cache[self.table_name]

        table_name = self.table_name

        class SessionModel(Base):
            __tablename__ = table_name
            __table_args__ = (
                Index(f"idx_{table_name}_session_id", "session_id"),
                Index(f"idx_{table_name}_user_id", "user_id"),
                Index(f"idx_{table_name}_session_status", "session_status"),
                Index(f"idx_{table_name}_session_start", "session_start"),
                {"extend_existing": True},
            )

            id = Column(Integer, primary_key=True, autoincrement=True)
            session_id = Column(String(255), nullable=False, unique=True)
            user_id = Column(String(255), nullable=False)
            app_name = Column(String(255), nullable=False)
            session_status = Column(String(50), nullable=False)  # active/ended/expired/error
            session_start = Column(DateTime, nullable=False)
            session_end = Column(DateTime, nullable=True)
            session_duration_seconds = Column(Float, nullable=True)

            # Stored as JSON text
            transcripts = Column(Text, nullable=True)
            user_metadata = Column(Text, nullable=True)
            session_metadata = Column(Text, nullable=True)
            voice_metadata = Column(Text, nullable=True)

            # Metrics
            error_count = Column(Integer, nullable=False, default=0)
            message_count = Column(Integer, nullable=False, default=0)
            average_response_time_ms = Column(Float, nullable=True)
            total_tokens_used = Column(Integer, nullable=False, default=0)

            # Timestamps — func.now() works on both MySQL and PostgreSQL
            created_at = Column(DateTime, nullable=False, server_default=func.now())
            updated_at = Column(
                DateTime,
                nullable=False,
                server_default=func.now(),
                onupdate=func.now()
            )

        _model_cache[self.table_name] = SessionModel
        return SessionModel

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def _ensure_table(self):
        """Create table if it doesn't exist."""
        engine = get_engine()
        Base.metadata.create_all(engine, tables=[self.Model.__table__])

    def _record_to_dict(self, record) -> Dict[str, Any]:
        """Convert a SQLAlchemy ORM record to a plain dict."""
        return {
            "id": record.id,
            "session_id": record.session_id,
            "user_id": record.user_id,
            "app_name": record.app_name,
            "session_status": record.session_status,
            "session_start": record.session_start.isoformat() if record.session_start else None,
            "session_end": record.session_end.isoformat() if record.session_end else None,
            "session_duration_seconds": record.session_duration_seconds,
            "transcripts": json.loads(record.transcripts) if record.transcripts else [],
            "user_metadata": json.loads(record.user_metadata) if record.user_metadata else {},
            "session_metadata": json.loads(record.session_metadata) if record.session_metadata else {},
            "voice_metadata": json.loads(record.voice_metadata) if record.voice_metadata else {},
            "error_count": record.error_count,
            "message_count": record.message_count,
            "average_response_time_ms": record.average_response_time_ms,
            "total_tokens_used": record.total_tokens_used,
            "created_at": record.created_at.isoformat() if record.created_at else None,
            "updated_at": record.updated_at.isoformat() if record.updated_at else None,
        }

    # -------------------------------------------------------------------------
    # CRUD operations
    # -------------------------------------------------------------------------

    def add_session_record(self, session_data_dict: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Insert a new session record.

        Args:
            session_data_dict: Session data from SessionData.to_dict()

        Returns:
            Dict with created record metadata, or None on error
        """
        self._ensure_table()
        engine, session = get_session()

        try:
            session_start = datetime.fromisoformat(session_data_dict["session_start"])
            session_end = None
            if session_data_dict.get("session_end"):
                session_end = datetime.fromisoformat(session_data_dict["session_end"])

            new_record = self.Model(
                session_id=session_data_dict["session_id"],
                user_id=session_data_dict["user_id"],
                app_name=session_data_dict["app_name"],
                session_status=session_data_dict["session_status"],
                session_start=session_start,
                session_end=session_end,
                session_duration_seconds=session_data_dict.get("session_duration_seconds"),
                transcripts=json.dumps(session_data_dict.get("transcripts", [])),
                user_metadata=json.dumps(session_data_dict.get("user_metadata", {})),
                session_metadata=json.dumps(session_data_dict.get("session_metadata", {})),
                voice_metadata=json.dumps(session_data_dict.get("voice_metadata", {})),
                error_count=session_data_dict.get("error_count", 0),
                message_count=session_data_dict.get("message_count", 0),
                average_response_time_ms=session_data_dict.get("average_response_time_ms"),
                total_tokens_used=session_data_dict.get("total_tokens_used", 0),
            )

            session.add(new_record)
            session.commit()

            logger.info(
                f"Session record saved: {session_data_dict['session_id']} "
                f"→ table '{self.table_name}'"
            )

            return {
                "id": new_record.id,
                "session_id": new_record.session_id,
                "user_id": new_record.user_id,
                "action": "created",
            }

        except Exception as e:
            session.rollback()
            logger.error(
                f"Error saving session {session_data_dict.get('session_id')}: {e}"
            )
            return None

        finally:
            session.close()

    def get_all_session_records(self) -> List[Dict[str, Any]]:
        """Retrieve all session records."""
        self._ensure_table()
        engine, session = get_session()

        try:
            records = session.query(self.Model).all()
            result = [self._record_to_dict(r) for r in records]
            logger.info(f"Retrieved {len(result)} session records from '{self.table_name}'")
            return result

        except Exception as e:
            logger.error(f"Error retrieving all session records: {e}")
            return []

        finally:
            session.close()

    def get_session_record_by_id(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a single session record by session_id."""
        self._ensure_table()
        engine, session = get_session()

        try:
            record = session.query(self.Model).filter(
                self.Model.session_id == session_id
            ).first()

            if not record:
                return None

            return self._record_to_dict(record)

        except Exception as e:
            logger.error(f"Error retrieving session {session_id}: {e}")
            return None

        finally:
            session.close()

    def get_sessions_by_user(self, user_id: str) -> List[Dict[str, Any]]:
        """Retrieve all session records for a specific user, newest first."""
        self._ensure_table()
        engine, session = get_session()

        try:
            records = (
                session.query(self.Model)
                .filter(self.Model.user_id == user_id)
                .order_by(self.Model.session_start.desc())
                .all()
            )
            result = [self._record_to_dict(r) for r in records]
            logger.info(f"Retrieved {len(result)} session records for user {user_id}")
            return result

        except Exception as e:
            logger.error(f"Error retrieving sessions for user {user_id}: {e}")
            return []

        finally:
            session.close()

    def delete_session_record(self, session_id: str) -> bool:
        """Delete a session record by session_id."""
        self._ensure_table()
        engine, session = get_session()

        try:
            record = session.query(self.Model).filter(
                self.Model.session_id == session_id
            ).first()

            if not record:
                logger.warning(f"No session record found for {session_id}")
                return False

            session.delete(record)
            session.commit()

            logger.info(f"Deleted session record: {session_id}")
            return True

        except Exception as e:
            session.rollback()
            logger.error(f"Error deleting session {session_id}: {e}")
            return False

        finally:
            session.close()


# Global singleton instance
session_repository = SessionRepository()