import os

import sys
import django
from django.apps import apps
from spel.scripts.config import database_app

def setup_django():
    # The Django project is importable as the top-level package `db`
    # (matching manage.py), so its parent directory must be on sys.path.
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "db.spel.settings")

    db_parent = str(database_app.resolve().parent)
    if db_parent not in sys.path:
        sys.path.insert(0, db_parent)

    if not apps.ready:
        django.setup()
    return

setup_django()
