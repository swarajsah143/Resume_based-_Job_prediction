"""
One-time migration script: SQLite (resumeai.db) -> MongoDB Atlas

Usage:
    1. Set MONGODB_URI in your .env file (or pass it as an environment variable)
    2. Run: python migrate_to_mongo.py
"""

import os
import sqlite3
from datetime import datetime, timezone

from dotenv import load_dotenv
from pymongo import MongoClient
from bson import ObjectId

load_dotenv()

SQLITE_PATH = os.path.join(os.path.dirname(__file__), 'resumeai.db')
MONGODB_URI = os.environ.get('MONGODB_URI', 'mongodb://localhost:27017/resumeai')


def parse_dt(value):
    """Parse a datetime string from SQLite into a timezone-aware datetime."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value
    for fmt in ('%Y-%m-%d %H:%M:%S.%f', '%Y-%m-%d %H:%M:%S', '%Y-%m-%dT%H:%M:%S.%f', '%Y-%m-%dT%H:%M:%S'):
        try:
            dt = datetime.strptime(value, fmt)
            return dt.replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    print(f"  WARNING: Could not parse datetime '{value}', storing as None")
    return None


def migrate():
    if not os.path.exists(SQLITE_PATH):
        print(f"SQLite database not found at {SQLITE_PATH}")
        return

    # Connect to SQLite
    conn = sqlite3.connect(SQLITE_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # Connect to MongoDB
    client = MongoClient(MONGODB_URI)
    db_name = MONGODB_URI.rsplit('/', 1)[-1].split('?')[0] or 'resumeai'
    mongo_db = client[db_name]

    print(f"Migrating from: {SQLITE_PATH}")
    print(f"Migrating to:   {MONGODB_URI}\n")

    # --- Migrate Users ---
    id_map = {}  # old integer ID -> new ObjectId
    cursor.execute('SELECT * FROM users')
    users = cursor.fetchall()

    for row in users:
        new_id = ObjectId()
        id_map[row['id']] = new_id
        mongo_db.users.insert_one({
            '_id': new_id,
            'fullname': row['fullname'],
            'email': row['email'],
            'password_hash': row['password_hash'],
            'auth_provider': row['auth_provider'] or 'local',
            'profile_picture': row['profile_picture'],
            'last_login': parse_dt(row['last_login']),
            'created_at': parse_dt(row['created_at']),
        })

    print(f"Users migrated: {len(users)}")

    # --- Migrate Analysis History ---
    cursor.execute('SELECT * FROM analysis_history')
    histories = cursor.fetchall()
    skipped = 0

    for row in histories:
        user_oid = id_map.get(row['user_id'])
        if not user_oid:
            skipped += 1
            continue
        mongo_db.analysis_history.insert_one({
            'user_id': str(user_oid),
            'filename': row['filename'],
            'predicted_role': row['predicted_role'],
            'confidence': row['confidence'],
            'resume_score': row['resume_score'],
            'created_at': parse_dt(row['created_at']),
        })

    print(f"Analysis history migrated: {len(histories) - skipped} (skipped {skipped} orphaned)")

    # --- Migrate OTP Verifications ---
    cursor.execute('SELECT * FROM otp_verifications')
    otps = cursor.fetchall()

    for row in otps:
        mongo_db.otp_verifications.insert_one({
            'email': row['email'],
            'otp': row['otp'],
            'fullname': row['fullname'],
            'password_hash': row['password_hash'],
            'expires_at': parse_dt(row['expires_at']),
            'is_verified': bool(row['is_verified']),
            'created_at': parse_dt(row['created_at']),
        })

    print(f"OTP verifications migrated: {len(otps)}")

    # --- Create indexes ---
    mongo_db.users.create_index('email', unique=True)
    mongo_db.analysis_history.create_index('user_id')
    mongo_db.analysis_history.create_index([('created_at', -1)])
    mongo_db.otp_verifications.create_index('email')
    mongo_db.otp_verifications.create_index([('created_at', -1)])
    print("\nIndexes created.")

    # --- Verify ---
    print(f"\nVerification:")
    print(f"  Users in MongoDB:    {mongo_db.users.count_documents({})}")
    print(f"  History in MongoDB:  {mongo_db.analysis_history.count_documents({})}")
    print(f"  OTPs in MongoDB:     {mongo_db.otp_verifications.count_documents({})}")

    conn.close()
    client.close()
    print("\nMigration complete! Keep resumeai.db as backup.")


if __name__ == '__main__':
    migrate()
