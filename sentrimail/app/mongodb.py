import json
import logging
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List

from pymongo import ASCENDING, MongoClient

logger = logging.getLogger(__name__)

MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
MONGODB_DB_NAME = os.getenv("MONGODB_DB_NAME", "sentrimail")

_client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=5000)
_mongo_db = _client[MONGODB_DB_NAME]

_DATA_DIR = Path(__file__).resolve().parents[1] / "data"
_DATA_DIR.mkdir(parents=True, exist_ok=True)


def _apply_projection(doc: Dict, projection: Dict | None) -> Dict:
    if not projection:
        return dict(doc)
    output = dict(doc)
    for key, include in projection.items():
        if include == 0 and key in output:
            output.pop(key, None)
    return output


def _matches_filter(doc: Dict, query: Dict | None) -> bool:
    if not query:
        return True
    for key, value in query.items():
        if doc.get(key) != value:
            return False
    return True


class _LocalCursor:
    def __init__(self, docs: List[Dict]):
        self._docs = docs

    def sort(self, key: str, direction: int):
        reverse = direction == -1
        self._docs.sort(key=lambda d: d.get(key, ""), reverse=reverse)
        return self

    def __iter__(self):
        return iter(self._docs)


class _LocalCollection:
    def __init__(self, file_path: Path):
        self.file_path = file_path
        if not self.file_path.exists():
            self.file_path.write_text("[]", encoding="utf-8")

    def _read(self) -> List[Dict]:
        try:
            return json.loads(self.file_path.read_text(encoding="utf-8"))
        except Exception:
            return []

    def _write(self, docs: List[Dict]) -> None:
        self.file_path.write_text(json.dumps(docs, ensure_ascii=True, indent=2), encoding="utf-8")

    def create_index(self, *args, **kwargs):
        return None

    def find(self, query=None, projection=None):
        docs = [_apply_projection(d, projection) for d in self._read() if _matches_filter(d, query)]
        return _LocalCursor(docs)

    def find_one(self, query=None, projection=None):
        for doc in self._read():
            if _matches_filter(doc, query):
                return _apply_projection(doc, projection)
        return None

    def insert_one(self, doc: Dict):
        docs = self._read()
        docs.append(dict(doc))
        self._write(docs)
        return SimpleNamespace(inserted_id=len(docs) - 1)

    def update_one(self, query: Dict, update: Dict, upsert: bool = False):
        docs = self._read()
        matched_index = None
        for i, doc in enumerate(docs):
            if _matches_filter(doc, query):
                matched_index = i
                break

        if matched_index is None:
            if not upsert:
                return SimpleNamespace(matched_count=0, modified_count=0, upserted_id=None)
            new_doc = dict(query)
            for k, v in update.get("$setOnInsert", {}).items():
                new_doc[k] = v
            for k, v in update.get("$set", {}).items():
                new_doc[k] = v
            docs.append(new_doc)
            self._write(docs)
            return SimpleNamespace(matched_count=0, modified_count=0, upserted_id=len(docs) - 1)

        target = docs[matched_index]
        for k, v in update.get("$set", {}).items():
            target[k] = v
        docs[matched_index] = target
        self._write(docs)
        return SimpleNamespace(matched_count=1, modified_count=1, upserted_id=None)


class _LocalDB:
    def __init__(self, data_dir: Path):
        self.users = _LocalCollection(data_dir / "users.json")
        self.complaints = _LocalCollection(data_dir / "complaints.json")
        self.login_logs = _LocalCollection(data_dir / "login_logs.json")


class _DBProxy:
    def __init__(self, backend):
        self._backend = backend

    def set_backend(self, backend):
        self._backend = backend

    def __getattr__(self, name):
        return getattr(self._backend, name)


db = _DBProxy(_mongo_db)


def init_mongodb() -> None:
    try:
        _client.admin.command("ping")
        db.set_backend(_mongo_db)
        logger.info("MongoDB connected: %s", MONGODB_URI)
    except Exception as exc:
        db.set_backend(_LocalDB(_DATA_DIR))
        logger.warning("MongoDB unavailable; using local JSON fallback storage: %s", exc)

    db.users.create_index([("username", ASCENDING)], unique=True)
    db.complaints.create_index([("username", ASCENDING)])
    db.complaints.create_index([("status", ASCENDING)])
    db.complaints.create_index([("created_at", ASCENDING)])
    db.login_logs.create_index([("username", ASCENDING), ("login_time", ASCENDING)])
