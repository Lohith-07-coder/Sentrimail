import os
from pymongo import ASCENDING, MongoClient


MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
MONGODB_DB_NAME = os.getenv("MONGODB_DB_NAME", "sentrimail")

_client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=5000)
db = _client[MONGODB_DB_NAME]


def init_mongodb() -> None:
    _client.admin.command("ping")

    db.users.create_index([("username", ASCENDING)], unique=True)
    db.complaints.create_index([("username", ASCENDING)])
    db.complaints.create_index([("status", ASCENDING)])
    db.complaints.create_index([("created_at", ASCENDING)])
    db.login_logs.create_index([("username", ASCENDING), ("login_time", ASCENDING)])
