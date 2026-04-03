#pgvector_service.py

"""
pgVector Service
uses: PostgreSQL + pgvector extension for vector storage and similarity search

"""
import os
import uuid
import json
import logging
import asyncio
from typing import List, Dict, Any, Optional

from dotenv import load_dotenv
from sqlalchemy import Column, Text, Index, text
from sqlalchemy.dialects.postgresql import UUID, JSONB
from pgvector.sqlalchemy import Vector
from google import genai
from google.genai.types import EmbedContentConfig

from app.repository.db_connector import get_engine, get_session
from app.schemas.base import Base

load_dotenv()

logger = logging.getLogger(__name__)

EMBEDDING_DIM = 768
_table_cache: Dict[str, Any] = {}


# =============================================================================
# EMBEDDING FUNCTION
# =============================================================================

class GeminiEmbeddingFunction:
    """
    Generates embeddings using Google Gemini (gemini-embedding-001, 768 dims).
    Uses the new google.genai SDK — consistent with agents.py and chromadb_service.py.
    """

    def __init__(
            self,
            model_name: str = "gemini-embedding-001",
            task_type: str = "RETRIEVAL_DOCUMENT",
            api_key: Optional[str] = None
    ):
        self.model_name = model_name
        self.task_type = task_type

        api_key = api_key or os.getenv("GOOGLE_API_KEY", "")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment variables")

        self.client = genai.Client(api_key=api_key, vertexai=False)

    def embed(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts.

        Args:
            texts: List of strings to embed

        Returns:
            List of embedding vectors (each 768 floats)
        """
        embeddings = []
        for text_item in texts:
            try:
                result = self.client.models.embed_content(
                    model=self.model_name,
                    contents=text_item,
                    config=EmbedContentConfig(
                        task_type=self.task_type,
                        output_dimensionality=EMBEDDING_DIM
                    )
                )
                embeddings.append(result.embeddings[0].values)
            except Exception as e:
                logger.error(f"Error generating embedding: {e}")
                embeddings.append([0.0] * EMBEDDING_DIM)
        return embeddings


# =============================================================================
# DYNAMIC TABLE MODEL
# =============================================================================

def _get_or_create_model(collection_name: str):
    """
    Dynamically create a SQLAlchemy model for a given collection (table).
    Uses a cache to avoid redefining tables across requests.

    Each collection maps to one PostgreSQL table with columns:
        id         UUID primary key
        document   TEXT  (the raw text chunk)
        metadata   JSONB (source URL, chunk index, etc.)
        embedding  vector(768)  ← pgvector column
    """
    if collection_name in _table_cache:
        return _table_cache[collection_name]

    class EmbeddingModel(Base):
        __tablename__ = collection_name
        __table_args__ = (
            Index(
                f"idx_{collection_name}_embedding",
                "embedding",
                postgresql_using="ivfflat",
                postgresql_ops={"embedding": "vector_cosine_ops"},
                postgresql_with={"lists": 100},
            ),
            {"extend_existing": True},
        )

        id = Column(
            UUID(as_uuid=True),
            primary_key=True,
            default=uuid.uuid4
        )
        document = Column(Text, nullable=False)
        metadata_ = Column("metadata", JSONB, nullable=True)
        embedding = Column(Vector(EMBEDDING_DIM), nullable=True)

    _table_cache[collection_name] = EmbeddingModel
    return EmbeddingModel


def _ensure_pgvector_extension(engine):
    """Enable pgvector extension if not already installed."""
    with engine.connect() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        conn.commit()


def _ensure_table(collection_name: str):
    """
    Enable pgvector extension and create the table if it doesn't exist.
    Called lazily before any read/write operation on a collection.
    """
    engine = get_engine()
    _ensure_pgvector_extension(engine)
    model = _get_or_create_model(collection_name)
    Base.metadata.create_all(engine, tables=[model.__table__])
    return model


# =============================================================================
# PGVECTOR SERVICE
# =============================================================================

class PgVectorService:
    """
    Drop-in replacement for ChromaService.
    Stores and queries document embeddings in PostgreSQL using pgvector.

    Public API matches the methods used by agents.py and other consumers:
        - query_collection(collection_name, query_text, n_results, filter_metadata)
        - get_or_create_collection(collection_name)   ← ensures table exists
        - add_chunked_documents(collection_name, documents)
        - add_chunked_documents_batch(collection_name, documents, batch_size)
        - get_all_collections()
        - get_collection_records(collection_name)
        - delete_records_from_collection(collection_name, record_ids)
        - remove_collection(collection_name)
    """

    # -------------------------------------------------------------------------
    # Collection management
    # -------------------------------------------------------------------------

    @staticmethod
    def get_or_create_collection(collection_name: str):
        """
        Ensure the table for this collection exists.
        Returns the SQLAlchemy model class (analogous to a ChromaDB collection object).
        """
        return _ensure_table(collection_name)

    @staticmethod
    def get_all_collections() -> Dict[str, Any]:
        """List all embedding tables in the database."""
        engine = get_engine()
        with engine.connect() as conn:
            rows = conn.execute(text(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema = 'public' AND table_type = 'BASE TABLE'"
            )).fetchall()

        # Filter to tables that look like embedding collections
        # (have an 'embedding' column)
        collections = []
        for (table_name,) in rows:
            with engine.connect() as conn:
                col_check = conn.execute(text(
                    "SELECT column_name FROM information_schema.columns "
                    "WHERE table_name = :t AND column_name = 'embedding'"
                ), {"t": table_name}).fetchone()
            if col_check:
                collections.append({"name": table_name, "metadata": {}})

        return {
            "status": "success",
            "collections_count": len(collections),
            "collections": collections
        }

    @staticmethod
    def remove_collection(collection_name: str) -> Dict[str, Any]:
        """Drop the table for this collection."""
        engine = get_engine()
        with engine.connect() as conn:
            conn.execute(text(f'DROP TABLE IF EXISTS "{collection_name}" CASCADE'))
            conn.commit()

        # Remove from cache
        _table_cache.pop(collection_name, None)

        return {
            "status": "success",
            "message": f"Collection '{collection_name}' deleted successfully",
            "collection_name": collection_name
        }

    # -------------------------------------------------------------------------
    # Document operations
    # -------------------------------------------------------------------------

    @staticmethod
    def get_collection_records(collection_name: str) -> Dict[str, Any]:
        """Retrieve all records from a collection."""
        model = _ensure_table(collection_name)
        engine, session = get_session()

        try:
            records = session.query(model).all()
            return {
                "status": "success",
                "collection_name": collection_name,
                "records_count": len(records),
                "records": {
                    "ids": [str(r.id) for r in records],
                    "documents": [r.document for r in records],
                    "metadatas": [r.metadata_ or {} for r in records]
                }
            }
        except Exception as e:
            logger.error(f"Error fetching records from '{collection_name}': {e}")
            raise
        finally:
            session.close()

    @staticmethod
    def delete_records_from_collection(
            collection_name: str,
            record_ids: List[str]
    ) -> Dict[str, Any]:
        """Delete specific records by UUID."""
        if not record_ids:
            raise ValueError("No record IDs provided")

        model = _ensure_table(collection_name)
        engine, session = get_session()

        try:
            uuids = [uuid.UUID(rid) for rid in record_ids]
            deleted = session.query(model).filter(
                model.id.in_(uuids)
            ).delete(synchronize_session=False)
            session.commit()

            return {
                "status": "success",
                "collection_name": collection_name,
                "records_deleted": deleted,
                "deleted_ids": record_ids
            }
        except Exception as e:
            session.rollback()
            logger.error(f"Error deleting records from '{collection_name}': {e}")
            raise
        finally:
            session.close()

    # -------------------------------------------------------------------------
    # Ingestion
    # -------------------------------------------------------------------------

    @staticmethod
    async def add_chunked_documents(
            collection_name: str,
            documents: List[str]
    ) -> Dict[str, Any]:
        """
        Add documents one-by-one, generating embeddings inline.
        Suitable for small batches (< 100 documents).
        """
        if not documents:
            raise ValueError("No documents provided")

        model = _ensure_table(collection_name)
        embedder = GeminiEmbeddingFunction(task_type="RETRIEVAL_DOCUMENT")
        engine, session = get_session()
        all_ids = []
        total = len(documents)

        logger.info(f"Starting upload of {total} documents to '{collection_name}'")

        try:
            for idx, doc_text in enumerate(documents, start=1):
                embedding = embedder.embed([doc_text])[0]
                record = model(
                    id=uuid.uuid4(),
                    document=doc_text,
                    metadata_={"source": str(uuid.uuid4())},
                    embedding=embedding
                )
                session.add(record)
                all_ids.append(str(record.id))

                if idx % 25 == 0 or idx == total:
                    session.commit()
                    logger.info(f"{idx}/{total} documents uploaded")

                if idx < total:
                    await asyncio.sleep(0.5)

            return {
                "status": "success",
                "collection_name": collection_name,
                "documents_added": len(all_ids),
                "document_ids": all_ids
            }
        except Exception as e:
            session.rollback()
            logger.error(f"Error adding documents to '{collection_name}': {e}")
            raise
        finally:
            session.close()

    @staticmethod
    async def add_chunked_documents_batch(
            collection_name: str,
            documents: List[str],
            batch_size: int = 50
    ) -> Dict[str, Any]:
        """
        Add documents in batches, generating embeddings per batch.
        More efficient for large document sets.
        """
        if not documents:
            raise ValueError("No documents provided")

        model = _ensure_table(collection_name)
        embedder = GeminiEmbeddingFunction(task_type="RETRIEVAL_DOCUMENT")
        engine, session = get_session()
        all_ids = []
        total = len(documents)

        logger.info(f"Starting batch upload of {total} documents to '{collection_name}'")

        try:
            for batch_start in range(0, total, batch_size):
                batch_docs = documents[batch_start:batch_start + batch_size]
                embeddings = embedder.embed(batch_docs)

                for doc_text, embedding in zip(batch_docs, embeddings):
                    record = model(
                        id=uuid.uuid4(),
                        document=doc_text,
                        metadata_={"source": str(uuid.uuid4())},
                        embedding=embedding
                    )
                    session.add(record)
                    all_ids.append(str(record.id))

                session.commit()
                uploaded = min(batch_start + batch_size, total)
                logger.info(f"{uploaded}/{total} documents uploaded")

                if batch_start + batch_size < total:
                    await asyncio.sleep(0.5)

            return {
                "status": "success",
                "collection_name": collection_name,
                "documents_added": len(all_ids),
                "document_ids": all_ids
            }
        except Exception as e:
            session.rollback()
            logger.error(f"Error in batch upload to '{collection_name}': {e}")
            raise
        finally:
            session.close()

    # -------------------------------------------------------------------------
    # Similarity search  ← core RAG operation
    # -------------------------------------------------------------------------

    @staticmethod
    def query_collection(
            collection_name: str,
            query_text: str,
            n_results: int = 5,
            filter_metadata: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Perform cosine similarity search using pgvector's <=> operator.

        Args:
            collection_name: Table name to search
            query_text:      Natural language query
            n_results:       Number of top results to return
            filter_metadata: Optional JSONB filter (e.g. {"source": "some-url"})
                             Applied as a PostgreSQL @> containment check.

        Returns:
            Dict with status, results list (document, metadata, distance)
        """
        try:
            model = _ensure_table(collection_name)
            engine, session = get_session()

            # Generate query embedding (RETRIEVAL_QUERY task type)
            embedder = GeminiEmbeddingFunction(task_type="RETRIEVAL_QUERY")
            query_embedding = embedder.embed([query_text])[0]

            try:
                # Build query with cosine distance ordering
                q = session.query(
                    model,
                    model.embedding.cosine_distance(query_embedding).label("distance")
                )

                # Optional metadata filter (JSONB containment)
                if filter_metadata:
                    q = q.filter(
                        model.metadata_.contains(filter_metadata)
                    )

                results = (
                    q.order_by("distance")
                    .limit(n_results)
                    .all()
                )

                formatted_results = [
                    {
                        "id": str(row.id),
                        "document": row.document,
                        "metadata": row.metadata_ or {},
                        "distance": float(distance)
                    }
                    for row, distance in results
                ]

                logger.info(
                    f"Query '{query_text[:60]}...' → "
                    f"{len(formatted_results)} results from '{collection_name}'"
                )

                return {
                    "status": "success",
                    "collection_name": collection_name,
                    "query": query_text,
                    "results_count": len(formatted_results),
                    "results": formatted_results
                }

            finally:
                session.close()

        except Exception as e:
            logger.error(f"Error querying collection '{collection_name}': {e}")
            raise Exception(f"Error querying collection: {str(e)}")


# =============================================================================
# CONNECTION TEST  (replaces test_chromadb_connection)
# =============================================================================

def test_pgvector_connection() -> Dict[str, Any]:
    """Test PostgreSQL + pgvector connectivity."""
    try:
        engine = get_engine()
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
            # Check pgvector extension
            result = conn.execute(text(
                "SELECT extname FROM pg_extension WHERE extname = 'vector'"
            )).fetchone()
            pgvector_installed = result is not None

        return {
            "status": "success",
            "message": "Connected to PostgreSQL successfully",
            "pgvector_installed": pgvector_installed,
            "postgres_url": os.getenv("POSTGRES_URL", "").split("@")[-1]  # hide credentials
        }
    except Exception as e:
        logger.error(f"pgVector connection test failed: {e}")
        return {
            "status": "failed",
            "message": f"Failed to connect: {str(e)}",
            "pgvector_installed": False
        }