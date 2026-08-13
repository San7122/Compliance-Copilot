import sys
from pathlib import Path

# Make `app` importable whether pytest is invoked from the repo root or from backend/.
# pytest only puts the test file's own directory on sys.path when tests/ has no
# __init__.py, which isn't enough to import the app package.
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Note: no heavyweight import shim is needed any more. Embeddings are produced by a
# hosted API client that is constructed lazily, so importing the app no longer drags in
# torch — which is also why the suite runs in about a second.
