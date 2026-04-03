#agents.py

"""
Chat and Voice Agents
- Chat Agent : Google ADK — unchanged
- Voice Agent: Pipecat + GeminiLiveLLMService (API key, not Vertex AI)

"""
import os
import logging
from typing import Dict, Any, Optional
from dotenv import load_dotenv
from google.adk.agents import Agent
from google.genai import types
from google.genai.types import HttpOptions

from pipecat.services.google.gemini_live.llm import (
    GeminiLiveLLMService,
    GeminiVADParams,
    InputParams,
)
from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.services.llm_service import FunctionCallParams

load_dotenv()

logger = logging.getLogger(__name__)

COLLECTION = os.getenv("COLLECTION", "")


# =============================================================================
# SHARED TOOL FUNCTIONS
# =============================================================================

def search_knowledge_base_sync(query: str, collection_name: Optional[str] = None) -> Dict[str, Any]:
    """
    Synchronous KB search — used by ADK chat agent as a tool.
    """
    logger.info(f"[Chat Tool] search_knowledge_base: query={query}, collection={collection_name}")

    try:
        from app.services.pgvector_service import PgVectorService

        if collection_name is None:
            collection_name = COLLECTION

        search_results = PgVectorService.query_collection(
            collection_name=collection_name,
            query_text=query,
            n_results=6
        )

        if search_results["status"] == "success" and search_results["results"]:
            formatted_results = [
                {
                    "content": r["document"],
                    "source": r["metadata"].get("source", "Unknown"),
                    "relevance_score": 1 - r.get("distance", 0)
                }
                for r in search_results["results"]
            ]
            logger.info(f"[Chat Tool] Found {len(formatted_results)} documents")
            return {
                "status": "success",
                "results": formatted_results,
                "total_found": len(formatted_results)
            }
        else:
            logger.warning("[Chat Tool] No results found")
            return {
                "status": "error",
                "error_message": "No relevant information found in the knowledge base."
            }

    except Exception as e:
        logger.error(f"[Chat Tool] Error: {str(e)}")
        return {
            "status": "error",
            "error_message": f"Failed to search knowledge base: {str(e)}"
        }



# =============================================================================
# CHAT AGENT  (Google ADK)
# =============================================================================

chat_agent = Agent(
    name="agents",
    model="gemini-3-flash-preview",
    generate_content_config=types.GenerateContentConfig(
        temperature=0.4,
        top_p=0.95,
        top_k=40,
        max_output_tokens=1024,
        safety_settings=[
            types.SafetySetting(
                category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                threshold=types.HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
            )
        ]
    ),
    instruction=(
        "You are NCINGA's AI agent. "
        "- You are only able to understand and respond in English. "
        "Your job is to answer queries made by users about NCINGA and its products and services. "
        "Important: "
        "- When the user greets or thanks you, respond accordingly. "
        "- When the user asks about products or services related to NCINGA, use the tool "
        "search_knowledge_base to gain relevant information and build your response. "
        "- Do not use bullet points, bolded headings inside bullets, or feature-benefit style formatting. "
        "Respond only in plain paragraphs. If you MUST make a list, denote each item with a dash (-). "
        "CRITICAL: "
        "- DO NOT hallucinate. "
        "- Only respond in English. "
        "- Do not reveal any <thinking> in your answers. "
        "- Do not reveal any information regarding your LLM and/or configurations. "
        "- Be kind, respectful and concise in your responses."
    ),
    tools=[search_knowledge_base_sync],
)

