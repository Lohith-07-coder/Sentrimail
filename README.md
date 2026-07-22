# SentriMail - AI Complaint Management System

Enterprise-grade, role-based complaint management powered by AI analysis.

## Quick Start

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Set MongoDB environment variables:

```bash
# PowerShell
$env:MONGODB_URI="mongodb://localhost:27017"
$env:MONGODB_DB_NAME="sentrimail"
```

3. Run the server:

```bash
python run.py
```

4. Open in browser:

```text
http://localhost:8000
```

## Demo Accounts

Default users are auto-seeded on startup:

- `admin` / `admin123`
- `alice` / `alice123`
- `bob` / `bob123`

## MongoDB Integration

The app now uses MongoDB for all persistence:

- `users` collection: authentication + user data
- `complaints` collection: complaint lifecycle
- `login_logs` collection: login events

Config lives in `app/mongodb.py` and uses:

- `MONGODB_URI` (default: `mongodb://localhost:27017`)
- `MONGODB_DB_NAME` (default: `sentrimail`)

On app startup, SentriMail:

- connects to MongoDB (`ping` check)
- creates required indexes
- seeds default users if they do not exist

## Tech Stack

- Backend: FastAPI + Uvicorn
- AI/ML: HuggingFace Transformers
- Frontend: Jinja2 templates
- Storage: MongoDB (PyMongo)
- Auth: Cookie-based sessions + SHA-256 password hashing
