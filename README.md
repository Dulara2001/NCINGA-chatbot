# NCINGA AI Chatbot

Demo: https://ds-enhanced-ncinga-chatbot-frontend-mvfrrnpiqq-as.a.run.app/

An AI-powered chat widget built with **FastAPI** (backend) and **React** (frontend), using **Google ADK** and **Gemini** for intelligent responses, **pgvector** for RAG-based knowledge retrieval, and deployed on **Google Cloud Run**.

---

## Features

- Streaming chat responses via Server-Sent Events (SSE)
- RAG-based knowledge base search using pgvector + Gemini embeddings
- Session management with PostgreSQL persistence
- Embeddable chat widget using Shadow DOM isolation
- Separate frontend and backend containers on Cloud Run
- CI/CD via GitHub Actions

---

## Tech Stack

### Backend
- **FastAPI** — REST API + SSE streaming
- **Google ADK** — AI agent framework
- **Google Gemini** — LLM (gemini-2.5-flash)
- **pgvector** — Vector similarity search
- **PostgreSQL** — Session storage
- **SQLAlchemy** — ORM
- **Cloud Run** — Serverless deployment

### Frontend
- **React + TypeScript** — Chat widget UI
- **Vite** — Build tool (IIFE library build)
- **Shadow DOM** — Style isolation for embeddable widget
- **Nginx** — Static file serving
- **Cloud Run** — Serverless deployment

---

## Project Structure

```
ds-website-chatbot/
├── .github/
│   └── workflows/
│       └── deploy.yml              # CI/CD — deploys both frontend and backend
├── app/
│   ├── api/
│   │   └── agent_api.py            # FastAPI routes (chat, session, SSE)
│   ├── repository/
│   │   ├── db_connector.py         # PostgreSQL connection
│   │   └── session_repository.py   # Session DB operations
│   ├── schemas/
│   │   └── base.py                 # SQLAlchemy base
│   ├── services/
│   │   ├── pgvector_service.py     # Vector search + embeddings
│   │   └── session_service.py      # Session management
│   ├── agents.py                   # ADK chat agent + tools
│   ├── main.py                     # FastAPI app entry point
│   └── Dockerfile                  # Backend container
├── frontend/
│   ├── src/
│   │   ├── api/
│   │   │   └── agent.ts            # API calls (startSession, streamMessage)
│   │   ├── components/
│   │   │   ├── ChatWidget.tsx      # Main chat UI component
│   │   │   └── ChatWidget.module.css
│   │   ├── types/
│   │   │   └── chat.ts             # TypeScript types
│   │   ├── App.tsx
│   │   └── main.tsx                # Widget mount (Shadow DOM)
│   ├── public/
│   │   └── ncinga.png              # Favicon
│   ├── index.html                  # Loads built IIFE bundle
│   ├── nginx.conf                  # Nginx config for Cloud Run
│   ├── vite.config.ts              # Vite library build config
│   └── Dockerfile                  # Frontend container
└── requirements.txt                # Python dependencies
```

---

## Local Development

### Prerequisites

- Python 3.13+
- Node.js 20+
- PostgreSQL with pgvector extension
- Google Cloud account with Gemini API access

### Backend

```bash
# Clone the repo and switch to the branch
git clone https://github.com/YOUR_ORG/YOUR_REPO.git
cd YOUR_REPO
git checkout enhanced-chatbot

# Create virtual environment
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Create .env file
cp .env.example .env
# Fill in your values (see Environment Variables section)

# Run the backend
uvicorn app.main:app --host 0.0.0.0 --port 8070 --reload
```

### Frontend

```bash
cd frontend

# Install dependencies
npm install

# Run the dev server
npm run dev
```

Frontend runs on `http://localhost:5173` and proxies `/agent` requests to `http://localhost:8070`.

---

## Environment Variables

Create a `.env` file in the project root:

```env
# PostgreSQL connection
# Local:
POSTGRES_URL=postgresql+psycopg2://user:password@localhost:5432/dbname

# Cloud Run (Cloud SQL socket):
# POSTGRES_URL=postgresql+psycopg2://user:password@/dbname?host=/cloudsql/project:region:instance

# Google AI
GOOGLE_API_KEY=your_gemini_api_key

# pgvector collection name
COLLECTION=your_collection_name

# Session table name
SESSION_TABLE_NAME=sessions
```

---

## Deployment

### Prerequisites

- Google Cloud project with billing enabled
- Cloud SQL (PostgreSQL) instance with pgvector extension
- Artifact Registry repository named `ncinga-chatbot`
- GitHub Actions secrets configured

### GitHub Secrets Required

| Secret | Description |
|--------|-------------|
| `GCP_SA_KEY` | Google Cloud service account key (JSON) |
| `GCP_PROJECT_ID` | Google Cloud project ID |
| `GCP_REGION` | Deployment region (e.g. `asia-south1`) |
| `CLOUD_SQL_CONNECTION_NAME` | e.g. `project:region:instance` |
| `POSTGRES_URL` | Cloud SQL socket connection string |
| `GOOGLE_API_KEY` | Gemini API key |
| `COLLECTION` | pgvector collection name |
| `SESSION_TABLE_NAME` | Sessions table name |
| `BACKEND_URL` | Cloud Run backend URL (set after first backend deploy) |

### CI/CD Flow

Pushing to the `enhanced-chatbot` branch automatically:

1. Builds and deploys the **backend** container to Cloud Run (`ncinga-backend`)
2. Waits for backend to finish
3. Builds and deploys the **frontend** container to Cloud Run (`ncinga-frontend`)

```
git push origin enhanced-chatbot
        ↓
GitHub Actions
        ↓
backend deploy → ncinga-backend-xxx.run.app
        ↓
frontend deploy → ncinga-frontend-xxx.run.app
```

### First-Time Deployment Steps

```bash
# 1. Enable Google Cloud APIs
gcloud services enable run.googleapis.com
gcloud services enable cloudbuild.googleapis.com
gcloud services enable sqladmin.googleapis.com
gcloud services enable artifactregistry.googleapis.com

# 2. Create Artifact Registry repository
gcloud artifacts repositories create ncinga-chatbot \
  --repository-format=docker \
  --location=YOUR_REGION

# 3. Create service account
gcloud iam service-accounts create github-actions-sa
gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
  --member="serviceAccount:github-actions-sa@YOUR_PROJECT_ID.iam.gserviceaccount.com" \
  --role="roles/run.admin"
gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
  --member="serviceAccount:github-actions-sa@YOUR_PROJECT_ID.iam.gserviceaccount.com" \
  --role="roles/artifactregistry.writer"
gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
  --member="serviceAccount:github-actions-sa@YOUR_PROJECT_ID.iam.gserviceaccount.com" \
  --role="roles/cloudsql.client"
gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
  --member="serviceAccount:github-actions-sa@YOUR_PROJECT_ID.iam.gserviceaccount.com" \
  --role="roles/iam.serviceAccountUser"

# 4. Download key and add to GitHub secrets
gcloud iam service-accounts keys create gcp-sa-key.json \
  --iam-account=github-actions-sa@YOUR_PROJECT_ID.iam.gserviceaccount.com
# Copy contents to GCP_SA_KEY secret, then delete
rm gcp-sa-key.json

# 5. Push to trigger deployment
git push origin enhanced-chatbot
```

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/agent/start-session` | Start a new chat session |
| `GET` | `/agent/chat/stream` | Stream chat response (SSE) |
| `POST` | `/agent/chat` | Non-streaming chat (legacy) |
| `GET` | `/agent/session/{id}` | Get session info |
| `DELETE` | `/agent/session/{id}` | End a session |
| `GET` | `/agent/health` | Health check |
| `GET` | `/swagger` | API documentation |

---

## Embedding the Widget

The frontend builds as an IIFE bundle that can be embedded in any webpage:

```html
<!-- Add to any HTML page -->
<script src="https://ncinga-frontend-xxx.run.app/ncinga-chat-widget.iife.js"></script>
```

The widget mounts itself automatically using Shadow DOM for style isolation.

---

## License

Internal use only — NCINGA © 2025
