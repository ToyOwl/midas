"""Vision-модальность: имена скриншотов, структурная валидация VLM-ответов."""
import json

from portagent.collect.charts import (_parse_chart_response, analyze_charts,
                                      scan_chart_drops)

UNIVERSE = ["MOEX:SBER", "US:AAPL", "US:MSFT"]


def test_scan_chart_drops_naming(tmp_path):
    directory = tmp_path / "2026-11-12"
    directory.mkdir(parents=True)
    (directory / "MOEX__SBER.png").write_bytes(b"png")
    (directory / "US__AAPL_daily.jpg").write_bytes(b"jpg")
    (directory / "random.txt").write_text("ignore", encoding="utf-8")
    (directory / "no_separator.png").write_bytes(b"png")
    drops = scan_chart_drops(tmp_path, "2026-11-12")
    assert [asset for asset, _ in drops] == ["MOEX:SBER", "US:AAPL"]


def test_parse_chart_response_valid():
    raw = json.dumps({"asset": "US:AAPL", "trend": "UP", "confidence": 1.0,
                      "note": "Пробой сопротивления"})
    label = _parse_chart_response(raw, expected_asset="US:AAPL", universe=UNIVERSE)
    assert label["trend"] == "up"
    assert label["confidence"] == 1.0          # clamp
    assert label["stance"] == 0.5              # up * confidence


def test_parse_chart_response_rejects_mismatch():
    raw = json.dumps({"asset": "US:MSFT", "trend": "down", "confidence": 0.9, "note": "x"})
    assert _parse_chart_response(raw, expected_asset="US:AAPL", universe=UNIVERSE) is None
    raw_bad_trend = json.dumps({"asset": "US:AAPL", "trend": "moon", "confidence": 0.9, "note": "x"})
    assert _parse_chart_response(raw_bad_trend, expected_asset="US:AAPL", universe=UNIVERSE) is None
    assert _parse_chart_response("not json", expected_asset="US:AAPL", universe=UNIVERSE) is None


def test_analyze_charts_degrades_per_image(tmp_path, monkeypatch):
    first, second = tmp_path / "a.png", tmp_path / "b.png"
    first.write_bytes(b"a-png-bytes")
    second.write_bytes(b"b-png-bytes")
    drops = [("US:AAPL", first), ("US:MSFT", second)]

    def fake_vision_chat(endpoint, model, image, mime, prompt, *, schema=None, timeout=120.0, max_tokens=1024, seed=20261112):
        if image.startswith(b"a"):
            return json.dumps({"asset": "US:AAPL", "trend": "down", "confidence": 0.8,
                               "note": "Пробой поддержки"}), {"latency_s": 1.0, "image_sha": "x"}
        raise RuntimeError("VLM недоступна")

    monkeypatch.setattr("portagent.collect.charts.vision_chat", fake_vision_chat)
    labels, statuses = analyze_charts(drops, universe=UNIVERSE,
                                      endpoint="http://127.0.0.1:8081",
                                      model="qwen", timeout=5)
    assert len(labels) == 1 and labels[0]["asset"] == "US:AAPL"
    assert labels[0]["stance"] == -0.4
    assert statuses[0]["ok"] and not statuses[1]["ok"]
