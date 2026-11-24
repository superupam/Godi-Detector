"""Tests for utils.load_model_meta — the metadata reader behind the UI stats."""

import json

import pytest

from utils import load_model_meta


def test_reads_the_keys_written_by_train_model(tmp_path):
    meta = {"accuracy": 97.8386, "dataset_size": 39095, "train_size": 31276, "test_size": 7819}
    path = tmp_path / "model_meta.json"
    path.write_text(json.dumps(meta), encoding="utf-8")

    assert load_model_meta(str(path)) == meta


def test_missing_file_returns_an_empty_dict(tmp_path):
    assert load_model_meta(str(tmp_path / "nope.json")) == {}


def test_malformed_json_returns_an_empty_dict(tmp_path):
    path = tmp_path / "model_meta.json"
    path.write_text("{not json,", encoding="utf-8")

    assert load_model_meta(str(path)) == {}


def test_json_that_is_not_an_object_returns_an_empty_dict(tmp_path):
    path = tmp_path / "model_meta.json"
    path.write_text(json.dumps([1, 2, 3]), encoding="utf-8")

    assert load_model_meta(str(path)) == {}


def test_empty_object_is_returned_as_is(tmp_path):
    path = tmp_path / "model_meta.json"
    path.write_text("{}", encoding="utf-8")

    assert load_model_meta(str(path)) == {}


def test_unreadable_path_returns_an_empty_dict(tmp_path):
    # A directory raises OSError on open, the same guard as a permission error.
    directory = tmp_path / "model_meta.json"
    directory.mkdir()

    assert load_model_meta(str(directory)) == {}


def test_default_path_is_the_tracked_metadata_file():
    with open("models/model_meta.json", "r", encoding="utf-8") as f:
        expected = json.load(f)

    meta = load_model_meta()
    assert meta["accuracy"] == expected["accuracy"]
    assert meta["dataset_size"] == expected["dataset_size"]
