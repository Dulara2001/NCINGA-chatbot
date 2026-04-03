#agent_api.py

"""
Unified Agent API
Handles both chat (ADK + SSE) and voice (Pipecat WebSocket) with session management.

Changes:
  - /chat/stream  : NEW SSE endpoint — streams ADK response token by token
  - stream_session: Updated to use create_pipeline() instead of run_voice_bot()
  - /chat         : Kept as-is for backwards compatibility
"""
import json
import logging
import uuid
import time
from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect, BackgroundTasks, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional
from google.adk.runners import Runner
from google.genai import types
from app.agents import chat_agent
from app.services.session_service import (
    session_manager, MessageRole, SessionType,
    SESSION_TIMEOUT_SECONDS, end_session_internal
)
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

agent_router = APIRouter(prefix="/agent", tags=["agent"])


# =============================================================================
# REQUEST / RESPONSE MODELS
# =============================================================================

class StartSessionResponse(BaseModel):
    status: str
    session_id: str
    user_id: str
    agent_name: str
    message: str


class ChatRequest(BaseModel):
    session_id: str
    message: str


class ChatResponse(BaseModel):
    status: str
    session_id: str
    response: str
    agent_name: str


class SessionInfoResponse(BaseModel):
    status: str
    session_id: str
    user_id: str
    is_active: bool
    message_count: int
    session_duration_seconds: Optional[float]


class EndSessionResponse(BaseModel):
    status: str
    session_id: str
    message: str
    message_count: int
    session_duration_seconds: Optional[float]


# =============================================================================
# SESSION ENDPOINTS 
# =============================================================================

@agent_router.post("/start-session", response_model=StartSessionResponse)
async def start_session():
    """Start a new chat/voice session."""
    try:
        session_id = str(uuid.uuid4())
        user_id = str(uuid.uuid4())

        logger.info(f"[Session {session_id}] Starting new session for user {user_id}")

        adk_session, session_data = await session_manager.create_session(
            session_id=session_id,
            user_id=user_id,
            app_name="website-chatbot",
            session_type=SessionType.CHAT,
            user_metadata={},
            session_metadata={"initial_mode": "chat"}
        )

        session_manager.add_message(
            session_id=session_id,
            role=MessageRole.SYSTEM,
            content="Chat session started",
            metadata={"event": "session_start"}
        )

        logger.info(f"[Session {session_id}] Session created successfully")

        return StartSessionResponse(
            status="success",
            session_id=session_id,
            user_id=user_id,
            agent_name=chat_agent.name,
            message=f"Session started successfully. You can now chat with {chat_agent.name}."
        )

    except Exception as e:
        logger.error(f"Error starting session: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start session: {str(e)}")


@agent_router.get("/session/{session_id}", response_model=SessionInfoResponse)
async def get_session_info(session_id: str):
    """Get information about a specific session."""
    try:
        session_data = session_manager.get_session_data(session_id)
        if not session_data:
            raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")

        return SessionInfoResponse(
            status="success",
            session_id=session_data.session_id,
            user_id=session_data.user_id,
            is_active=(session_data.session_status.value == "active"),
            message_count=session_data.message_count,
            session_duration_seconds=session_data.session_duration_seconds
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[Session {session_id}] Error getting session info: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get session info: {str(e)}")


@agent_router.delete("/session/{session_id}", response_model=EndSessionResponse)
async def end_session(session_id: str, background_tasks: BackgroundTasks):
    """
    End a chat session immediately and process cleanup in background.
    Called when user closes chat or clicks refresh.
    """
    try:
        session_data = session_manager.get_session_data(session_id)
        if not session_data:
            raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")

        message_count = session_data.message_count
        duration = session_data.session_duration_seconds or 0

        logger.info(f"[Session {session_id}] User requested end. Messages: {message_count}")

        session_manager.pause_expiration_timer(session_id)
        session_manager.end_session(session_id)

        background_tasks.add_task(end_session_internal, session_id, "user_request")

        return EndSessionResponse(
            status="success",
            session_id=session_id,
            message="Session ended successfully. Cleanup in progress.",
            message_count=message_count,
            session_duration_seconds=duration
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[Session {session_id}] Error ending session: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to end session: {str(e)}")


# =============================================================================
# CHAT  —  blocking (kept for backwards compatibility)
# =============================================================================

@agent_router.post("/chat", response_model=ChatResponse)
async def chat(chat_request: ChatRequest):
    """
    Send a chat message and receive the full response at once.
    Kept for backwards compatibility — prefer /chat/stream for new frontends.
    """
    try:
        session_id = chat_request.session_id
        message = chat_request.message

        logger.info(f"[Session {session_id}] Chat message: {message[:100]}...")

        session_data = session_manager.get_session_data(session_id)
        if not session_data:
            raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")

        session_service = session_manager.get_adk_session_service(session_id)
        if not session_service:
            raise HTTPException(status_code=404, detail=f"Session service for '{session_id}' not found")

        session_manager.reset_expiration_timer(session_id, timeout_seconds=SESSION_TIMEOUT_SECONDS)

        session_manager.add_message(
            session_id=session_id,
            role=MessageRole.USER,
            content=message,
            metadata={"source": "chat"}
        )

        runner = Runner(
            agent=chat_agent,
            app_name="website-chatbot",
            session_service=session_service
        )

        content = types.Content(role="user", parts=[types.Part(text=message)])

        start_time = time.time()
        agent_response_text = "No response received."

        async for event in runner.run_async(
                user_id=session_data.user_id,
                session_id=session_id,
                new_message=content
        ):
            if event.is_final_response():
                agent_response_text = event.content.parts[0].text
                break

        response_time_ms = (time.time() - start_time) * 1000

        session_manager.add_message(
            session_id=session_id,
            role=MessageRole.AGENT,
            content=agent_response_text,
            metadata={"source": "chat", "response_time_ms": response_time_ms}
        )

        session_manager.update_performance_metrics(
            session_id=session_id,
            response_time_ms=response_time_ms,
            tokens_used=None
        )

        logger.info(f"[Session {session_id}] Chat response. Time: {response_time_ms:.2f}ms")

        return ChatResponse(
            status="success",
            session_id=session_id,
            response=agent_response_text,
            agent_name=chat_agent.name
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[Session {chat_request.session_id}] Chat error: {e}")
        session_manager.add_message(
            session_id=chat_request.session_id,
            role=MessageRole.ERROR,
            content=f"Chat error: {str(e)}",
            metadata={"error_type": type(e).__name__}
        )
        raise HTTPException(status_code=500, detail=f"Error processing chat: {str(e)}")


# =============================================================================
# CHAT STREAM  —  SSE (Google ADK + Server-Sent Events)
# =============================================================================

@agent_router.get("/chat/stream")
async def chat_stream(session_id: str, message: str):
    """
    Stream chat response token-by-token using SSE (Server-Sent Events).

    The ADK runner.run_async() loop already yields events incrementally —
    this endpoint wraps those events into an SSE stream so the frontend
    can render tokens as they arrive instead of waiting for the full response.

    Frontend usage:
        const es = new EventSource(
            `/agent/chat/stream?session_id=${sid}&message=${encodeURIComponent(msg)}`
        )
        es.onmessage = (e) => {
            if (e.data === '[DONE]') { es.close(); return; }
            const chunk = JSON.parse(e.data);
            if (chunk.error) { showError(chunk.error); es.close(); return; }
            appendToUI(chunk.text);
        }

    Event format:
        data: {"text": "Hello! ", "done": false}   ← token chunk
        data: [DONE]                                ← stream finished
        data: {"error": "..."}                      ← error (followed by [DONE])
    """
    # Validate session before opening stream
    session_data = session_manager.get_session_data(session_id)
    if not session_data:
        raise HTTPException(
            status_code=404,
            detail=f"Session '{session_id}' not found. Please start a new session."
        )

    async def event_generator():
        try:
            session_service = session_manager.get_adk_session_service(session_id)
            if not session_service:
                yield f"data: {json.dumps({'error': 'Session service not found'})}\n\n"
                yield "data: [DONE]\n\n"
                return

            # Reset inactivity timer on each message
            session_manager.reset_expiration_timer(
                session_id,
                timeout_seconds=SESSION_TIMEOUT_SECONDS
            )

            # Log user message to transcript
            session_manager.add_message(
                session_id=session_id,
                role=MessageRole.USER,
                content=message,
                metadata={"source": "chat_stream"}
            )

            # Build ADK runner
            runner = Runner(
                agent=chat_agent,
                app_name="website-chatbot",
                session_service=session_service
            )

            content = types.Content(
                role="user",
                parts=[types.Part(text=message)]
            )

            start_time = time.time()
            full_response = ""

            # Iterate ADK events and forward text chunks as SSE
            async for event in runner.run_async(
                user_id=session_data.user_id,
                session_id=session_id,
                new_message=content
            ):
                # Partial text chunks — stream them immediately
                if (
                    event.content
                    and event.content.parts
                    and not event.is_final_response()
                ):
                    for part in event.content.parts:
                        if hasattr(part, "text") and part.text:
                            full_response += part.text
                            yield f"data: {json.dumps({'text': part.text, 'done': False})}\n\n"

                # Final response
                if event.is_final_response():
                    final_text = (
                        event.content.parts[0].text
                        if event.content and event.content.parts
                        else ""
                    )
                    # Fallback: if nothing was streamed yet, send full response now
                    if not full_response and final_text:
                        full_response = final_text
                        yield f"data: {json.dumps({'text': final_text, 'done': False})}\n\n"
                    break

            response_time_ms = (time.time() - start_time) * 1000

            # Log full response to transcript
            session_manager.add_message(
                session_id=session_id,
                role=MessageRole.AGENT,
                content=full_response,
                metadata={
                    "source": "chat_stream",
                    "response_time_ms": response_time_ms
                }
            )

            session_manager.update_performance_metrics(
                session_id=session_id,
                response_time_ms=response_time_ms,
                tokens_used=None
            )

            logger.info(
                f"[Session {session_id}] Stream complete. "
                f"Time: {response_time_ms:.2f}ms, "
                f"Chars: {len(full_response)}"
            )

            yield "data: [DONE]\n\n"

        except Exception as e:
            logger.error(f"[Session {session_id}] SSE stream error: {e}")
            session_manager.add_message(
                session_id=session_id,
                role=MessageRole.ERROR,
                content=f"Stream error: {str(e)}",
                metadata={"error_type": type(e).__name__}
            )
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
            yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",      # disables nginx buffering — critical for SSE
            "Connection": "keep-alive",
        }
    )



# =============================================================================
# HEALTH
# =============================================================================

@agent_router.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "Agent API",
        "active_sessions": len(session_manager.get_active_sessions()),
        "chat": "google-adk-sse",
    }