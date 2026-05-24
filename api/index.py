"""
Vercel Serverless Function Entry Point

This file imports the Flask app from the project root and exposes it
as `app` for Vercel's Python runtime. Vercel looks for an `app` (WSGI)
object in /api/index.py by default.
"""

import sys
import os

# Ensure the project root is on the Python path so we can import app.py
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import app  # noqa: E402
