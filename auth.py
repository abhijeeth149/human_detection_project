"""
auth.py — Mock credential store for Vidyaksha login system.
Replace with a real DB lookup in production.
"""

"""
auth.py — JSON-based credential store for Vidyaksha login system.
Supports registering new students dynamically.
"""

import json
import os

CREDENTIALS_FILE = "credentials.json"

# Default fallback if file doesn't exist
DEFAULT_DATA = {
    "faculty": {
        "admin": "vidya123",
        "faculty1": "teach2024"
    },
    "student": {
        "std001": "pass123",
        "std002": "pass456"
    }
}

def _load_credentials() -> dict:
    if not os.path.exists(CREDENTIALS_FILE):
        _save_credentials(DEFAULT_DATA)
        return DEFAULT_DATA
    try:
        with open(CREDENTIALS_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return DEFAULT_DATA

def _save_credentials(data: dict):
    with open(CREDENTIALS_FILE, "w") as f:
        json.dump(data, f, indent=4)

# ── Verify Functions ──────────────────────────────────────────────────────────
def verify_faculty(employee_id: str, password: str) -> bool:
    """Return True if the Employee ID and password match a Faculty record."""
    creds = _load_credentials()
    return creds.get("faculty", {}).get(employee_id.strip()) == password

def verify_student(reg_no: str, password: str) -> bool:
    """Return True if the Registration Number and password match a Student record."""
    creds = _load_credentials()
    return creds.get("student", {}).get(reg_no.strip()) == password

def register_student(reg_no: str, password: str) -> bool:
    """Adds a new student credential to the JSON store. Returns False if already exists."""
    creds = _load_credentials()
    reg_clean = reg_no.strip()
    
    # Check if student already exists
    if reg_clean in creds.get("student", {}):
        return False
        
    creds["student"][reg_clean] = password
    _save_credentials(creds)
    return True
