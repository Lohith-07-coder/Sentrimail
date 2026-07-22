"""
SentriMail — Entry Point
Run with: python run.py
"""

import uvicorn
import os
import sys

# Make console output resilient on Windows code pages (cp1252, etc.)
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Ensure the project root is in the path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

# Create data directory with absolute path
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
os.makedirs(DATA_DIR, exist_ok=True)

if __name__ == "__main__":
    print("\n" + "="*50)
    try:
        print("  🛡️  SentriMail — AI Complaint Management")
    except UnicodeEncodeError:
        print("  SentriMail - AI Complaint Management")
    print("="*50)
    print(f" -> Running on port {os.environ.get('PORT', 8000)}")
    print("  → Admin:  admin / admin123")
    print("  → User:   alice / alice123  |  bob / bob123")
    print("="*50 + "\n")

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8000)),
        reload=False,
        log_level="info"
    )
