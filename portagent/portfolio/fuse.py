"""Слияние модальностей-видов: SNF (нелинейно, с гейтом качества) или линейная смесь.

Вид — матрица расстояний D на (под)множестве активов. Базовый вид «price»
покрывает все активы; aux-виды (sentiment, legal, tech) — только покрытые.
aux-вид участвует в слиянии, если выполнены ОБА условия:
- покрытие >= fusion.min_coverage;
- качество (consistency, см. features.view_quality) >= fusion.snf_min_quality
  (только для SNF: исторический тест показал, что шумовой вид разрушает
  кросс-диффузию; линейная смесь робастна и гейтом не фильтруется).
Вес вида в SNF = его качество. Непокрытые пары деградируют до цен.
Один живой вид -> он и есть результат (mode=single).
"""
from __future__ import annotations

from typing import Any

import pandas as pd

from .snf import snf


def _align_to_base(D_aux: pd.DataFrame, D_base: pd.DataFrame) -> pd.DataFrame:
    """Aux-вид на полный набор активов; пропуски (NaN / отсутствующие активы)
    заполняем значениями базового вида — парный fallback к ценам."""
    assets = list(D_base.index)
    aligned = D_aux.reindex(index=assets, columns=assets)
    return aligned.where(aligned.notna(), D_base)


def fuse_distances(views: dict[str, pd.DataFrame | None],
                   *, base: str = "price", method: str = "snf",
                   coverage: dict[str, float] | None = None,
                   view_quality: dict[str, float] | None = None,
                   w_price: float = 0.7, min_coverage: float = 0.6,
                   snf_neighbors: int = 15, snf_iterations: int = 20,
                   snf_min_quality: float = 0.3) -> dict[str, Any]:
    """Слияние видов расстояний.

    views: имя вида -> матрица расстояний (None/пустая = вид отсутствует).
    coverage: имя вида -> доля покрытия активами (для aux-видов).
    view_quality: имя вида -> consistency вида (для гейта/весов SNF);
      отсутствующее качество у aux-вида трактуется как 0 (не пускать).
    Возвращает {D, mode, note, views_used}.
    """
    if base not in views or views[base] is None or views[base].empty:
        raise ValueError(f"базовый вид {base!r} обязателен")
    D_base = views[base]
    coverage = coverage or {}
    quality = view_quality or {}

    def covered(name: str, D) -> bool:
        return D is not None and not D.empty and coverage.get(name, 1.0) >= min_coverage

    candidates = [name for name in views if name != base and covered(name, views[name])]
    if method == "snf":
        admitted = [name for name in candidates
                    if quality.get(name, 0.0) >= snf_min_quality]
    else:
        admitted = candidates  # linear робастен — гейт не нужен
    rejected = [name for name in views if name != base and name not in admitted]

    if not admitted:
        reasons = []
        for name in rejected:
            if not covered(name, views[name]):
                reasons.append(f"{name} (покрытие)")
            else:
                value = quality.get(name)
                reasons.append(f"{name} (качество {value if value is None else round(value, 2)})")
        return {
            "D": D_base,
            "mode": "single",
            "views_used": [base],
            "note": (f"Слияние не выполнялось: доступен только вид «{base}»"
                     + (f"; отклонены: {', '.join(reasons)}" if reasons else "") + "."),
        }

    aligned: dict[str, pd.DataFrame] = {base: D_base}
    for name in admitted:
        aligned[name] = _align_to_base(views[name], D_base)

    if method == "linear":
        aux_weight = (1.0 - w_price) / len(admitted)
        fused = w_price * D_base.copy()
        for name in admitted:
            fused = fused + aux_weight * aligned[name]
        return {
            "D": fused,
            "mode": "linear",
            "views_used": [base] + admitted,
            "note": (f"Линейная смесь: {w_price:.0%} «{base}» + {aux_weight:.0%} на вид "
                     f"({', '.join(admitted)}); пары без покрытия aux-вида деградируют к ценам."),
        }

    weights = {base: 1.0}
    weights.update({name: float(quality.get(name, 1.0)) for name in admitted})
    fused = snf(aligned, k=snf_neighbors, t=snf_iterations, weights=weights)
    quality_note = ", ".join(f"{name}={weights[name]:.2f}" for name in admitted)
    return {
        "D": fused,
        "mode": "snf",
        "views_used": [base] + admitted,
        "note": (f"SNF (k={snf_neighbors}, T={snf_iterations}, веса=качество: "
                 f"{quality_note}) сливает виды {', '.join([base] + admitted)}; "
                 + (f"отклонены гейтом: {', '.join(rejected)}; " if rejected else "")
                 + "непокрытые aux-пары деградируют к ценам."),
    }
