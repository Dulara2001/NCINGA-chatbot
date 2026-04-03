#session_service.py

"""
Comprehensive Session Service for managing chat and voice sessions
Tracks sessions, transcripts, and provides utilities for session management
"""
import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict, fields
from enum import Enum
from google.adk.sessions import InMemorySessionService
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

SESSION_TIMEOUT_SECONDS = 300


class MessageRole(Enum):
    """Enum for message roles in transcript"""
    SYSTEM = "system"
    AGENT = "agent"
    USER = "user"
    ERROR = "error"


class SessionType(Enum):
    """Enum for session types"""
    CHAT = "chat"
    VOICE = "voice"


class SessionStatus(Enum):
    """Enum for session status"""
    ACTIVE = "active"
    ENDED = "ended"
    EXPIRED = "expired"
    ERROR = "error"


@dataclass
class TranscriptMessage:
    """
    Represents a single message in the transcript
    """
    role: MessageRole
    content: str
    timestamp: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with properly escaped content"""
        return {
            "role": self.role.value,
            "content": json.dumps(self.content),  # Properly escape quotes and special chars
            "timestamp": self.timestamp,
            "metadata": self.metadata
        }

    def to_formatted_string(self) -> str:
        """Convert to formatted string for display"""
        escaped_content = self.content.replace('"', '\\"')
        return f'{self.role.value}: "{escaped_content}"'


@dataclass
class SessionData:
    """
    Comprehensive session data structure
    """
    session_id: str
    user_id: str
    app_name: str
    session_type: SessionType
    session_start: str  # ISO format timestamp
    session_end: Optional[str] = None  # ISO format timestamp
    session_duration_seconds: Optional[float] = None
    session_status: SessionStatus = SessionStatus.ACTIVE
    transcripts: List[TranscriptMessage] = field(default_factory=list)
    expiration_task: Optional[asyncio.Task] = field(default=None, init=False, repr=False, compare=False)

    # Additional useful metadata
    user_metadata: Dict[str, Any] = field(default_factory=dict)  # User info, IP, device, etc.
    session_metadata: Dict[str, Any] = field(default_factory=dict)  # Custom session data
    error_count: int = 0
    message_count: int = 0

    # Voice-specific metadata
    voice_metadata: Dict[str, Any] = field(default_factory=dict)  # Audio quality, interruptions, etc.

    # Performance metrics
    average_response_time_ms: Optional[float] = None
    total_tokens_used: int = 0

    def calculate_duration(self) -> Optional[float]:
        """Calculate session duration in seconds"""
        if self.session_start and self.session_end:
            start = datetime.fromisoformat(self.session_start)
            end = datetime.fromisoformat(self.session_end)
            self.session_duration_seconds = (end - start).total_seconds()
            return self.session_duration_seconds
        return None

    def add_transcript(
            self,
            role: MessageRole,
            content: str,
            metadata: Optional[Dict[str, Any]] = None
    ):
        """Add a message to the transcript"""
        message = TranscriptMessage(
            role=role,
            content=content,
            timestamp=datetime.now(timezone.utc).isoformat(),
            metadata=metadata or {}
        )
        self.transcripts.append(message)
        self.message_count += 1

        if role == MessageRole.ERROR:
            self.error_count += 1

    def get_formatted_transcript(self) -> str:
        """Get full transcript as formatted string"""
        return "\n".join([msg.to_formatted_string() for msg in self.transcripts])

    def get_transcript_json(self) -> List[Dict[str, Any]]:
        """Get transcript as list of dictionaries"""
        return [msg.to_dict() for msg in self.transcripts]

    def to_dict(self) -> Dict[str, Any]:
        """Convert session data to dictionary for serialization"""
        # Get all fields except those marked with compare=False
        data = {}
        for field_info in fields(self):
            if field_info.name != 'expiration_task':  # Explicitly exclude task
                data[field_info.name] = getattr(self, field_info.name)

        data['session_type'] = self.session_type.value
        data['session_status'] = self.session_status.value
        data['transcripts'] = self.get_transcript_json()
        return data

    def end_session(self):
        """Mark session as ended and calculate duration"""
        self.session_end = datetime.now(timezone.utc).isoformat()
        self.session_status = SessionStatus.ENDED
        self.calculate_duration()


class SessionManager:
    """
    Global session manager for handling all sessions
    Manages InMemorySessionService instances and custom session tracking
    """

    def __init__(self):
        # Store ADK InMemorySessionService for each session
        self.adk_session_services: Dict[str, InMemorySessionService] = {}

        # Store ADK session objects
        self.adk_sessions: Dict[str, Any] = {}

        # Store our custom session data
        self.session_data: Dict[str, SessionData] = {}

        logger.info("SessionManager initialized")

    def _create_expiration_task(self, session_id: str, timeout_seconds: int = SESSION_TIMEOUT_SECONDS):
        """Create background task to expire session after timeout"""

        async def expire_session():
            try:
                await asyncio.sleep(timeout_seconds)
                logger.info(f"[Session {session_id}] Inactivity timeout reached, expiring session")

                # Get session data
                session_data = self.get_session_data(session_id)
                if session_data and session_data.session_status == SessionStatus.ACTIVE:
                    # Mark as expired
                    session_data.session_status = SessionStatus.EXPIRED
                    session_data.session_end = datetime.now(timezone.utc).isoformat()
                    session_data.calculate_duration()

                    # Save to database and cleanup
                    await end_session_internal(session_id, "inactivity_timeout")

            except asyncio.CancelledError:
                logger.debug(f"[Session {session_id}] Expiration task cancelled")
                raise
            except Exception as e:
                logger.error(f"[Session {session_id}] Error in expiration task: {e}")

        task = asyncio.create_task(expire_session())
        return task

    def reset_expiration_timer(self, session_id: str, timeout_seconds: int = SESSION_TIMEOUT_SECONDS):
        """Cancel existing timer and create new one"""
        session_data = self.get_session_data(session_id)
        if not session_data:
            return False

        # Cancel existing task
        if session_data.expiration_task and not session_data.expiration_task.done():
            session_data.expiration_task.cancel()
            logger.debug(f"[Session {session_id}] Expiration timer reset")

        # Create new task
        session_data.expiration_task = self._create_expiration_task(session_id, timeout_seconds)
        return True

    def pause_expiration_timer(self, session_id: str):
        """Pause expiration timer (for voice chat)"""
        session_data = self.get_session_data(session_id)
        if not session_data:
            return False

        # Cancel task
        if session_data.expiration_task and not session_data.expiration_task.done():
            session_data.expiration_task.cancel()
            session_data.expiration_task = None
            logger.debug(f"[Session {session_id}] Expiration timer paused")

        return True

    def resume_expiration_timer(self, session_id: str, timeout_seconds: int = SESSION_TIMEOUT_SECONDS):
        """Resume expiration timer (after voice chat ends)"""
        return self.reset_expiration_timer(session_id, timeout_seconds)



    async def create_session(
            self,
            session_id: str,
            user_id: str,
            app_name: str,
            session_type: SessionType,
            user_metadata: Optional[Dict[str, Any]] = None,
            session_metadata: Optional[Dict[str, Any]] = None
    ) -> tuple[Any, SessionData]:
        """
        Create a new session with both ADK session service and custom tracking

        Args:
            session_id: Unique session identifier
            user_id: User identifier
            app_name: Application name
            session_type: Type of session (CHAT or VOICE)
            user_metadata: Optional user metadata (IP, device, location, etc.)
            session_metadata: Optional custom session metadata

        Returns:
            Tuple of (ADK session object, SessionData object)
        """
        try:
            # Create ADK session service
            session_service = InMemorySessionService()

            # Create ADK session
            adk_session = await session_service.create_session(
                app_name=app_name,
                user_id=user_id,
                session_id=session_id,
            )

            # Store ADK session service and session
            self.adk_session_services[session_id] = session_service
            self.adk_sessions[session_id] = adk_session

            # Create custom session data
            session_data = SessionData(
                session_id=session_id,
                user_id=user_id,
                app_name=app_name,
                session_type=session_type,
                session_start=datetime.now(timezone.utc).isoformat(),
                user_metadata=user_metadata or {},
                session_metadata=session_metadata or {}
            )

            # Add initial system message
            session_data.add_transcript(
                role=MessageRole.SYSTEM,
                content=f"Session started - Type: {session_type.value}",
                metadata={"event": "session_start"}
            )

            # Store session data
            self.session_data[session_id] = session_data
            session_data.expiration_task = self._create_expiration_task(session_id, SESSION_TIMEOUT_SECONDS)

            logger.info(f"Session created: {session_id} (User: {user_id}, Type: {session_type.value})")

            return adk_session, session_data

        except Exception as e:
            logger.error(f"Error creating session {session_id}: {e}")
            raise

    def get_adk_session(self, session_id: str) -> Optional[Any]:
        """Get ADK session object"""
        return self.adk_sessions.get(session_id)

    def get_adk_session_service(self, session_id: str) -> Optional[InMemorySessionService]:
        """Get ADK session service"""
        return self.adk_session_services.get(session_id)

    def get_session_data(self, session_id: str) -> Optional[SessionData]:
        """Get custom session data"""
        return self.session_data.get(session_id)

    def add_message(
            self,
            session_id: str,
            role: MessageRole,
            content: str,
            metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Add a message to session transcript

        Args:
            session_id: Session identifier
            role: Message role (SYSTEM, AGENT, USER, ERROR)
            content: Message content
            metadata: Optional message metadata

        Returns:
            True if successful, False otherwise
        """
        session_data = self.get_session_data(session_id)
        if session_data:
            session_data.add_transcript(role, content, metadata)
            logger.debug(f"Message added to session {session_id}: {role.value}")
            return True
        else:
            logger.warning(f"Session {session_id} not found")
            return False

    def end_session(self, session_id: str) -> bool:
        """
        End a session and calculate final metrics

        Args:
            session_id: Session identifier

        Returns:
            True if successful, False otherwise
        """
        session_data = self.get_session_data(session_id)
        if session_data:
            session_data.end_session()
            session_data.add_transcript(
                role=MessageRole.SYSTEM,
                content="Session ended",
                metadata={"event": "session_end"}
            )
            logger.info(
                f"Session ended: {session_id} "
                f"(Duration: {session_data.session_duration_seconds}s, "
                f"Messages: {session_data.message_count})"
            )
            return True
        else:
            logger.warning(f"Session {session_id} not found")
            return False

    def delete_session(self, session_id: str) -> bool:
        """
        Delete a session and clean up all associated data

        Args:
            session_id: Session identifier

        Returns:
            True if session was deleted, False if not found
        """
        # End session if still active
        session_data = self.get_session_data(session_id)
        if session_data and session_data.session_status == SessionStatus.ACTIVE:
            self.end_session(session_id)

        # Remove all session data
        adk_service = self.adk_session_services.pop(session_id, None)
        adk_session = self.adk_sessions.pop(session_id, None)
        custom_data = self.session_data.pop(session_id, None)

        deleted = any([adk_service, adk_session, custom_data])

        if deleted:
            logger.info(f"Session deleted: {session_id}")
        else:
            logger.warning(f"Session not found for deletion: {session_id}")

        return deleted

    def get_all_sessions(self) -> Dict[str, SessionData]:
        """Get all session data"""
        return self.session_data

    def get_active_sessions(self) -> Dict[str, SessionData]:
        """Get all active sessions"""
        return {
            sid: data for sid, data in self.session_data.items()
            if data.session_status == SessionStatus.ACTIVE
        }

    def get_session_count(self) -> Dict[str, int]:
        """Get session counts by status"""
        counts = {
            "total": len(self.session_data),
            "active": 0,
            "ended": 0,
            "expired": 0,
            "error": 0
        }

        for session in self.session_data.values():
            if session.session_status == SessionStatus.ACTIVE:
                counts["active"] += 1
            elif session.session_status == SessionStatus.ENDED:
                counts["ended"] += 1
            elif session.session_status == SessionStatus.EXPIRED:
                counts["expired"] += 1
            elif session.session_status == SessionStatus.ERROR:
                counts["error"] += 1

        return counts

    def get_session_summary(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a summary of session data

        Args:
            session_id: Session identifier

        Returns:
            Dictionary with session summary or None if not found
        """
        session_data = self.get_session_data(session_id)
        if not session_data:
            return None

        return {
            "session_id": session_data.session_id,
            "user_id": session_data.user_id,
            "session_type": session_data.session_type.value,
            "session_status": session_data.session_status.value,
            "session_start": session_data.session_start,
            "session_end": session_data.session_end,
            "duration_seconds": session_data.session_duration_seconds,
            "message_count": session_data.message_count,
            "error_count": session_data.error_count,
            "total_tokens_used": session_data.total_tokens_used,
        }

    def export_session_to_json(self, session_id: str) -> Optional[str]:
        """
        Export session data as JSON string

        Args:
            session_id: Session identifier

        Returns:
            JSON string or None if session not found
        """
        session_data = self.get_session_data(session_id)
        if session_data:
            return json.dumps(session_data.to_dict(), indent=2)
        return None

    def export_all_sessions_to_json(self) -> str:
        """
        Export all session data as JSON string

        Returns:
            JSON string of all sessions
        """
        all_sessions = {
            sid: data.to_dict()
            for sid, data in self.session_data.items()
        }
        return json.dumps(all_sessions, indent=2)

    def cleanup_ended_sessions(self, keep_last_n: int = 100) -> int:
        """
        Clean up old ended sessions, keeping only the most recent ones

        Args:
            keep_last_n: Number of most recent ended sessions to keep

        Returns:
            Number of sessions deleted
        """
        ended_sessions = [
            (sid, data) for sid, data in self.session_data.items()
            if data.session_status == SessionStatus.ENDED
        ]

        if len(ended_sessions) <= keep_last_n:
            return 0

        # Sort by end time (oldest first)
        ended_sessions.sort(key=lambda x: x[1].session_end or "")

        # Delete oldest sessions
        sessions_to_delete = ended_sessions[:-keep_last_n]
        deleted_count = 0

        for session_id, _ in sessions_to_delete:
            if self.delete_session(session_id):
                deleted_count += 1

        logger.info(f"Cleaned up {deleted_count} old ended sessions")
        return deleted_count

    def update_session_metadata(
            self,
            session_id: str,
            metadata_updates: Dict[str, Any]
    ) -> bool:
        """
        Update session metadata

        Args:
            session_id: Session identifier
            metadata_updates: Dictionary of metadata to update

        Returns:
            True if successful, False otherwise
        """
        session_data = self.get_session_data(session_id)
        if session_data:
            session_data.session_metadata.update(metadata_updates)
            logger.debug(f"Updated metadata for session {session_id}")
            return True
        return False

    def update_performance_metrics(
            self,
            session_id: str,
            response_time_ms: Optional[float] = None,
            tokens_used: Optional[int] = None
    ) -> bool:
        """
        Update performance metrics for a session

        Args:
            session_id: Session identifier
            response_time_ms: Response time in milliseconds
            tokens_used: Number of tokens used

        Returns:
            True if successful, False otherwise
        """
        session_data = self.get_session_data(session_id)
        if not session_data:
            return False

        if response_time_ms is not None:
            # Calculate running average
            if session_data.average_response_time_ms is None:
                session_data.average_response_time_ms = response_time_ms
            else:
                # Weighted average
                count = session_data.message_count
                current_avg = session_data.average_response_time_ms
                session_data.average_response_time_ms = (
                        (current_avg * (count - 1) + response_time_ms) / count
                )

        if tokens_used is not None:
            session_data.total_tokens_used += tokens_used

        return True


# Global session manager instance
session_manager = SessionManager()


async def end_session_internal(session_id: str, reason: str = "user_request"):
    """
    Internal function to end session and save to database

    Args:
        session_id: Session identifier
        reason: Reason for ending session
    """

    from app.repository.session_repository import session_repository

    try:
        # Get session data before ending
        session_data = session_manager.get_session_data(session_id)
        if not session_data:
            logger.warning(f"[Session {session_id}] Session not found for ending")
            return

        # End the session
        session_manager.end_session(session_id)

        # Get updated session data
        session_data = session_manager.get_session_data(session_id)

        # Save to database
        session_dict = session_data.to_dict()
        result = session_repository.add_session_record(session_dict)

        if result:
            logger.info(
                f"[Session {session_id}] Session saved to database. "
                f"Reason: {reason}, Messages: {session_data.message_count}, "
                f"Duration: {session_data.session_duration_seconds}s"
            )
        else:
            logger.error(f"[Session {session_id}] Failed to save session to database")

        # Delete session from memory
        session_manager.delete_session(session_id)

        logger.info(f"[Session {session_id}] Session cleanup completed")

    except Exception as e:
        logger.error(f"[Session {session_id}] Error ending session: {e}")


# Helper functions for easy access
async def create_session(
        session_id: str,
        user_id: str,
        app_name: str,
        session_type: SessionType,
        user_metadata: Optional[Dict[str, Any]] = None,
        session_metadata: Optional[Dict[str, Any]] = None
) -> tuple[Any, SessionData]:
    """Helper function to create a session"""
    return await session_manager.create_session(
        session_id, user_id, app_name, session_type, user_metadata, session_metadata
    )


def get_session(session_id: str) -> Optional[SessionData]:
    """Helper function to get session data"""
    return session_manager.get_session_data(session_id)


def add_message(
        session_id: str,
        role: MessageRole,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
) -> bool:
    """Helper function to add a message to transcript"""
    return session_manager.add_message(session_id, role, content, metadata)


def end_session(session_id: str) -> bool:
    """Helper function to end a session"""
    return session_manager.end_session(session_id)


def delete_session(session_id: str) -> bool:
    """Helper function to delete a session"""
    return session_manager.delete_session(session_id)


def get_session_summary(session_id: str) -> Optional[Dict[str, Any]]:
    """Helper function to get session summary"""
    return session_manager.get_session_summary(session_id)


def export_session(session_id: str) -> Optional[str]:
    """Helper function to export session as JSON"""
    return session_manager.export_session_to_json(session_id)
