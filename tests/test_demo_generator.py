"""The public demo must never create synthetic materials in working data."""
from scripts import make_demo_fixture


def test_generated_materials_stay_in_demo(tmp_path, monkeypatch):
    monkeypatch.setattr(make_demo_fixture, "PROJECT_ROOT", tmp_path)
    make_demo_fixture.main(tmp_path / "fixtures" / "demo")
    assert not (tmp_path / "data").exists()
    assert len(list((tmp_path / ".demo" / "data" / "manual" / "charts").rglob("*.png"))) == 12
    assert len(list((tmp_path / ".demo" / "data" / "manual" / "courts").rglob("*.json"))) == 1
