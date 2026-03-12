import os

import sys
import django
from django.apps import apps
from spel.scripts.config import database_app

def setup_django():
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "spel.db.spel.settings")

    sys.path.insert(0, str(database_app.resolve()))

    if not apps.ready:
        django.setup()
    return

setup_django()
