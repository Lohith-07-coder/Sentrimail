# SentriMail - AI Complaint Management System

Enterprise-grade, role-based complaint management platform powered by hybrid AI sentiment, emotion, priority analysis, and automated resolution workflows.

---

## 🏗️ Project Architecture & Layout

SentriMail follows a clean, modular, layered architecture separating core configuration, persistence, domain services, web controllers, and automated tests.

```text
Sentrimail/
├── app/
│   ├── main.py                 # FastAPI application factory & middleware initialization
│   ├── core/                   # Core configuration & infrastructure
│   │   ├── config.py           # Application settings (Pydantic Settings)
│   │   ├── database.py         # MongoDB driver with local JSON fallback storage
│   │   ├── logging.py          # Structured logging setup
│   │   └── security.py         # Password hashing & JWT session security
│   ├── schemas/                # Pydantic data schemas
│   │   ├── admin.py
│   │   ├── complaint.py
│   │   └── user.py
│   ├── repositories/           # Persistence boundary (Data Access Layer)
│   │   ├── complaint_repository.py
│   │   └── user_repository.py
│   ├── services/               # Core Domain & Business Logic
│   │   ├── ai_service.py       # Sentiment, emotion, priority & response generation pipeline
│   │   ├── auth_service.py     # User authentication & session management
│   │   ├── complaint_service.py# Complaint lifecycle, auto-backfill, escalation & analytics
│   │   ├── email_service.py    # SMTP resolution email notifications
│   │   └── transcription_service.py # Audio transcription via OpenAI Whisper
│   └── routers/                # FastAPI APIRouters (Controllers)
│       ├── admin.py            # Admin portal dashboard, complaint triage & CSV export
│       ├── api.py              # Programmatic REST API endpoints
│       ├── auth.py             # Login, register, logout & session refresh
│       └── user.py             # Customer portal, complaint submission & tracking
├── data/                       # Local storage & ML dataset models
├── templates/                  # Jinja2 HTML templates
├── static/                     # CSS stylesheets & frontend assets
├── tests/                      # Automated pytest verification suite
│   └── test_app.py
├── run.py                      # Application launcher script
├── migrate.py                  # Database migration utility
├── requirements.txt            # Dependency specifications
└── README.md                   # Project documentation
```

---

## ⚡ Features

- **Hybrid AI Analysis Engine**: Combines HuggingFace Transformers (DistilBERT, DistilRoBERTa) with rule-based fallback for sentiment, emotion, root cause, and priority scoring (`LOW`, `MEDIUM`, `HIGH`, `CRITICAL`).
- **Automated Resolution & Smart Suggestions**: Auto-handles low-urgency safe complaints and generates contextual responses for admins.
- **Multilingual Support**: Auto-detects complaint language and provides transparent translation via Google Translator.
- **Audio Complaint Transcription**: Transcribes voice complaints using OpenAI Whisper.
- **Role-Based Access Control (RBAC)**: Secure cookie-based JWT sessions separating User and Admin portals.
- **Persistence Fallback**: Connects seamlessly to MongoDB or automatically falls back to local JSON persistence if MongoDB is offline.
- **Background Escalation**: Periodically escalates pending complaints based on time thresholds using `APScheduler`.

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Environment Variables (Optional)

Configure MongoDB connection details via environment variables (or rely on defaults/local JSON fallback):

```bash
# PowerShell
$env:MONGODB_URI="mongodb://localhost:27017"
$env:MONGODB_DB_NAME="sentrimail"
```

### 3. Run the Server

```bash
python run.py
```

### 4. Access the Application

Open your browser and navigate to:
```text
http://localhost:8000
```

---

## 🔑 Demo Accounts

Default users are automatically seeded on initial application startup:

| Role  | Username | Password   |
| ----- | -------- | ---------- |
| Admin | `admin`  | `admin123` |
| User  | `alice`  | `alice123` |
| User  | `bob`    | `bob123`   |

---

## 🧪 Running Automated Tests

Run the unit and integration test suite using `pytest`:

```bash
python -m pytest tests/test_app.py
```

---

## 🛠️ Tech Stack

- **Backend**: FastAPI, Uvicorn, Pydantic v2
- **AI/ML**: HuggingFace Transformers (DistilBERT, DistilRoBERTa, Flan-T5), OpenAI Whisper
- **Translation & NLP**: `langdetect`, `deep-translator`
- **Database & Storage**: MongoDB (PyMongo), Local JSON File Storage
- **Authentication**: PyJWT, Passlib (SHA-256 / Bcrypt)
- **Background Tasks & Scheduling**: `APScheduler`
- **Testing**: `pytest`, `httpx`
