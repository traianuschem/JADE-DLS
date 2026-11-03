"""Integrity check of committed test fixtures against tests/data/manifest.json."""

import hashlib
import json
import os

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def _entries():
    with open(os.path.join(DATA, "manifest.json"), encoding="utf-8") as fh:
        return json.load(fh)


def test_manifest_lists_files():
    assert len(_entries()) >= 6


def test_fixture_files_match_manifest_sha256():
    for e in _entries():
        path = os.path.join(DATA, *e["file"].split("/"))
        with open(path, "rb") as fh:
            b = fh.read()
        assert len(b) == e["size"], e["file"]
        assert hashlib.sha256(b).hexdigest() == e["sha256"], e["file"]


def test_alv_fixture_dir_has_no_unlisted_files():
    listed = {e["file"] for e in _entries() if e["file"].startswith("alv_mono/")}
    on_disk = {"alv_mono/" + f for f in os.listdir(os.path.join(DATA, "alv_mono"))}
    assert on_disk == listed
