"""Versioned prompt templates, one file per agent per version
(``a1_evidence_v1.py``, and so on). Bumping a prompt means adding a new
``_v2`` file and pointing the agent module at it -- the old version stays
importable so cached results keyed on it remain meaningful, and a
prompt change invalidates only its own cache entries (see
``agents/base.py``'s cache key)."""
