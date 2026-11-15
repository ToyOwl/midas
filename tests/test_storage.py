"""Raw-архив: контент-адресуемые имена, ничего не перезаписывается."""
from portagent.storage.raw import archive, write_manifest


def test_same_payload_reused(tmp_path):
    first = archive(tmp_path, "2026-11-12", "moex_SBER.json", b"payload")
    second = archive(tmp_path, "2026-11-12", "moex_SBER.json", b"payload")
    assert first["path"] == second["path"]
    assert second["reused"] is True


def test_changed_payload_new_file_keeps_old(tmp_path):
    first = archive(tmp_path, "2026-11-12", "yf_AAPL.csv", b"v1")
    second = archive(tmp_path, "2026-11-12", "yf_AAPL.csv", b"v2")
    assert first["path"] != second["path"]
    assert second["reused"] is False
    import pathlib

    files = [p.name for p in (tmp_path / "2026-11-12").iterdir() if p.suffix == ".csv"]
    assert len(files) == 2  # обе версии сохранены


def test_manifest_written(tmp_path):
    entry = archive(tmp_path, "2026-11-12", "x.json", b"data")
    path = write_manifest(tmp_path, "2026-11-12", "market", [entry], {"source": "test"})
    assert path.exists() and "market" in path.name
