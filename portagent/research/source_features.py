"""Point-in-time features; no LLM/provider dependency in the training process."""
import json
from pathlib import Path
import pandas as pd


def features_for(rows, events):
    out = pd.DataFrame(0., index=rows.index, columns=["tg_count", "tg_negative", "tg_positive", "tg_channels", "court_pressure", "legal_news_pressure"])
    out["tg_missing"] = 1.
    out["court_missing"] = 1.
    out["legal_news_missing"] = 1.
    for idx, row in rows.iterrows():
        matched = [e for e in events if e["ticker"] == row["asset"] and e["available_date"] <= row["date"]]
        # At a decision date use the newest observed version of each message.
        matched = list({(e["kind"], e["news_id"]): e for e in sorted(matched, key=lambda e: (e["available_date"], e["analyzed_at"]))}.values())
        sources = set()
        for event in matched:
            age = (pd.Timestamp(row["date"]) - pd.Timestamp(event["available_date"])).days
            if age > 120:
                continue
            kind = event["kind"]
            decay = .5 ** (age / (5 if kind == "telegram" else 20))
            negative = max(0, -event["sentiment"]) * event["confidence"] * decay
            if kind == "telegram":
                out.loc[idx, "tg_count"] += decay
                out.loc[idx, "tg_negative"] += negative
                out.loc[idx, "tg_positive"] += max(0, event["sentiment"]) * event["confidence"] * decay
                out.loc[idx, "tg_missing"] = 0
                sources.add(event["source"])
            elif kind == "court":
                out.loc[idx, "court_pressure"] += negative
                out.loc[idx, "court_missing"] = 0
            else:
                out.loc[idx, "legal_news_pressure"] += negative
                out.loc[idx, "legal_news_missing"] = 0.
        out.loc[idx, "tg_channels"] = len(sources)
    return out
