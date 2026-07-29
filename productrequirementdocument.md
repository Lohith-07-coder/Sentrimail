# Product Requirements Document (PRD)
## SentriMail — Enterprise AI Complaint Management System

**Document status:** Reverse-engineered from repository documentation, for learning/rebuild purposes
**Source repo:** github.com/Lohith-07-coder/Sentrimail
**Version:** 1.0

---

## 1. Problem Statement

Organizations receive support complaints across many channels, languages, and severity levels. Traditional helpdesk systems fail in three ways:

- **Delayed triage** — critical issues (legal threats, outages, security incidents) get buried in a queue alongside routine requests.
- **Manual overhead** — agents spend time drafting responses to low-urgency, repetitive queries that could be resolved automatically.
- **Inconsistent prioritization** — urgency is assigned by human judgment rather than a consistent, auditable scoring method.
- **Language friction** — non-English complaints require manual translation before an agent can act.

## 2. Goals

| Goal | Description |
|---|---|
| G1 | Automatically classify incoming complaints by sentiment, emotion, and urgency |
| G2 | Compute a consistent, explainable priority score (0–100) for every complaint |
| G3 | Auto-resolve safe, low-priority complaints without human involvement |
| G4 | Escalate unresolved complaints over time based on SLA thresholds |
| G5 | Support complaints in any language and via voice input |
| G6 | Remain operational even if the primary database is unreachable |
| G7 | Give admins a real-time view of complaint volume, category, and priority |

### Non-goals (explicitly out of scope for v1)
- Taking automated actions beyond drafting a response (e.g., issuing refunds)
- Real-time push notifications (WebSockets)
- Third-party SSO (Google/Microsoft login)
- Large-scale vector database infrastructure (Qdrant/Pinecone)

## 3. Target Users & Roles

| Role | Description | Key needs |
|---|---|---|
| **Customer (`user`)** | Submits complaints via text or voice, in any language | Fast submission, ability to track status via a complaint code, resolution in their own language |
| **Support Admin (`admin`)** | Reviews, triages, and resolves complaints | Dashboard sorted by priority, AI-suggested responses, analytics, override control |

## 4. User Stories

1. As a customer, I can submit a complaint in my native language and receive a response in that same language.
2. As a customer, I can upload a voice recording instead of typing, and have it transcribed automatically.
3. As a customer, I can look up my complaint's status using a public tracking code without logging in.
4. As an admin, I see complaints sorted by an automatically computed priority so I address the most urgent ones first.
5. As an admin, I receive an AI-drafted response I can edit and send, rather than writing from scratch.
6. As an admin, if a ticket sits unresolved past its SLA window, I want it automatically escalated so nothing falls through the cracks.
7. As an admin, I want a dashboard showing category breakdown, priority distribution, and average response time.
8. As a system operator, I want the app to keep working even if MongoDB is down, using local storage as a fallback.

## 5. Functional Requirements

### 5.1 Complaint Ingestion
- Accept complaint submission via web form (text) and audio upload (WAV/MP3).
- Detect input language automatically; translate to English internally for analysis.
- Transcribe audio to text using a speech-to-text model before analysis.

### 5.2 AI Analysis Pipeline
- Run sentiment classification (positive/negative/neutral with confidence score).
- Run emotion classification (anger, fear, sadness, disgust, surprise, joy, neutral).
- Scan for emergency/urgency keywords (e.g., "lawsuit," "police," "data loss") that force an immediate priority override.
- Generate a root-cause summary describing the likely underlying issue.

### 5.3 Priority Engine
- Compute a composite priority score (0–100) from sentiment intensity, emotion severity, urgency keywords, and complaint length.
- Map score to severity band: LOW (0–24), MEDIUM (25–49), HIGH (50–74), CRITICAL (75–100).
- Automatically re-escalate a ticket's severity if it remains unresolved past a time-based SLA threshold.

### 5.4 Auto-Resolution
- For LOW-priority, auto-resolvable complaints: attempt to match against a curated response dataset using text-similarity search.
- If no strong match, fall back to a generated template response.
- Reject weak/generic matches for HIGH and CRITICAL tickets — those always route to a human.

### 5.5 Admin Workflow
- Dashboard listing complaints with priority, category, status, and timestamps.
- Ability to view AI-suggested response and send a final (possibly edited) response via email.
- Analytics: category counts, priority distribution, daily volume trend, average response time, CSV export.

### 5.6 Auth & Access Control
- Username/password login with hashed password storage.
- Session managed via JWT stored in an HTTP-only cookie.
- Two roles: `user` and `admin`, with route-level access separation.

### 5.7 Data Persistence
- Primary storage in MongoDB.
- Automatic, transparent fallback to local JSON file storage if MongoDB is unreachable, with no change in application behavior or downtime.

## 6. Non-Functional Requirements

| Category | Requirement |
|---|---|
| **Availability** | App must remain functional without an active MongoDB connection |
| **Performance** | Rule-based analysis path should respond in the low tens of milliseconds; transformer-based path under ~200ms |
| **Security** | Passwords hashed (not reversible); JWT in HTTP-only, SameSite cookies to prevent XSS token theft; all inputs validated before persistence |
| **Scalability** | Should sustain several hundred requests/sec on modest hardware using the rule-based fallback path |
| **Resilience** | Every AI component (sentiment, emotion, response generation) must have a non-ML rule-based fallback so the system degrades gracefully rather than failing |
| **Internationalization** | Must support complaint submission and resolution in 100+ languages |

## 7. Success Metrics

- % of LOW-priority complaints auto-resolved without human touch
- Average time from submission to first response, by priority tier
- % of CRITICAL complaints correctly flagged by keyword/emotion detection (vs. missed)
- System uptime / successful fallback rate when MongoDB is unavailable
- Admin time saved (proxy: number of AI-suggested responses sent with no or minor edits)

## 8. System Constraints & Assumptions

- Runs as a single FastAPI service (no microservices split in v1).
- ML models run locally/in-process (HuggingFace pipelines, Whisper `base`) — no external paid inference API required.
- Assumes moderate traffic; not designed for massive horizontal scale in this version (see "Future Improvements" for scale-oriented roadmap items).

## 9. Roadmap / Future Improvements (post-v1)

- Migrate response-matching from TF-IDF to a vector database (Qdrant/Pinecone) for scale.
- LLM agent tool-calling to take real actions (e.g., issuing refunds), not just drafting text.
- WebSocket-based real-time push of CRITICAL alerts to admins.
- OAuth2 / SSO login (Google, Microsoft).

---

### How to use this PRD for learning
Treat each numbered functional requirement (5.1–5.7) as a milestone. Build them in roughly that order, starting with a minimal version of 5.2 (priority engine only, no ML) before layering in transformers, translation, auth, and persistence. This mirrors how the original system's fallback-first design philosophy suggests it was likely built.