"""Fresh stock TabM + NGBoost Normal + XGBoostLSS Gaussian.

The event bank belongs to the fold encoder. C3 adapts the original nested
interval calibration to nested absolute tails >=1.5 and >=2.5 sigma.
"""
from __future__ import annotations

import copy
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar
from scipy.special import expit, logit, softmax
from scipy.stats import norm

CANDIDATES = ("tabm", "tabm_kl", "ngboost_c3", "xgboostlss_c3", "tabm_ngboost_z", "tabm_both_z")
SEED = 20261114


class Preprocess:
    def fit(self, x):
        x = np.asarray(x, float)
        self.median = np.nanmedian(x, axis=0)
        self.median = np.nan_to_num(self.median)
        filled = np.where(np.isfinite(x), x, self.median)
        self.mean, self.std = filled.mean(axis=0), filled.std(axis=0)
        self.std[self.std < 1e-8] = 1
        return self

    def transform(self, x):
        x = np.asarray(x, float)
        x = np.where(np.isfinite(x), x, self.median)
        return np.clip((x - self.mean) / self.std, -10, 10).astype("float32")


def fit_tabm(x, y, vx, vy, *, epochs=40):
    import torch
    from tabm import TabM
    torch.manual_seed(SEED)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TabM.make(n_num_features=x.shape[1], d_out=2, k=8, n_blocks=2, d_block=64, dropout=.1).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.002, weight_decay=.0003)
    xx, yy = torch.as_tensor(x, device=device), torch.as_tensor(y, dtype=torch.long, device=device)
    vxx, vyy = torch.as_tensor(vx, device=device), torch.as_tensor(vy, dtype=torch.long, device=device)
    best_loss, best, best_epoch, stale = float("inf"), None, 0, 0
    rng = np.random.default_rng(SEED)
    for epoch in range(epochs):
        model.train()
        for indices in np.array_split(rng.permutation(len(x)), max(1, int(np.ceil(len(x) / 256)))):
            logits = model(xx[indices])
            loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), yy[indices].repeat_interleave(logits.shape[1]))
            optimizer.zero_grad(); loss.backward(); optimizer.step()
        model.eval()
        with torch.inference_mode():
            logits = torch.cat([model(part) for part in vxx.split(512)])
            value = float(torch.nn.functional.nll_loss(logits.softmax(-1).mean(1).clamp_min(1e-8).log(), vyy))
        if value < best_loss - 1e-6:
            best_loss, best, best_epoch, stale = value, copy.deepcopy(model.state_dict()), epoch + 1, 0
        else:
            stale += 1
        if stale >= 7:
            break
    model.load_state_dict(best); model.eval()
    return model, {"epoch": best_epoch, "validation_logloss": best_loss, "device": device}


def tabm_predict(model, x):
    import torch
    device = next(model.parameters()).device
    probabilities, hidden = [], []
    with torch.inference_mode():
        for part in np.array_split(x, max(1, int(np.ceil(len(x) / 512)))):
            tensor = torch.as_tensor(part, device=device)
            probabilities.append(model(tensor).softmax(-1).mean(1)[:, 1].cpu().numpy())
            hidden.append(model.backbone(model.ensemble_view(tensor)).cpu().numpy())
    return np.concatenate(probabilities), np.concatenate(hidden)


def kl_scores(hidden, bank):
    if len(bank) < 25:
        raise ValueError("Недостаточно событий в обучающем банке")
    ref = softmax(bank.astype(float), axis=2)
    lr = np.log(ref.clip(1e-15))
    scores = []
    for q in softmax(hidden.astype(float), axis=2):
        distances = .5 * ((ref - q) * (lr - np.log(q.clip(1e-15)))).sum(2).mean(1)
        scores.append(-float(np.partition(distances, 24)[:25].mean()))
    return np.asarray(scores)


def fit_families(train, valid, features, target, *, epochs=40):
    from ngboost import NGBRegressor
    from ngboost.distns import Normal
    from sklearn.tree import DecisionTreeRegressor
    import xgboost as xgb
    from xgboostlss.model import XGBoostLSS
    from xgboostlss.distributions.Gaussian import Gaussian
    prep = Preprocess().fit(train[features])
    x, vx = prep.transform(train[features]), prep.transform(valid[features])
    tabm, audit = fit_tabm(x, train.event.to_numpy(int), vx, valid.event.to_numpy(int), epochs=epochs)
    print("TabM", audit, flush=True)
    ngb = NGBRegressor(Dist=Normal, Base=DecisionTreeRegressor(max_depth=3, min_samples_leaf=30, random_state=SEED),
        n_estimators=80, learning_rate=.05, natural_gradient=True, verbose=False, random_state=SEED)
    y, vy = train.target_z.to_numpy(float), valid.target_z.to_numpy(float)
    ngb.fit(x, y, X_val=vx, Y_val=vy, early_stopping_rounds=12)
    iterations = (int(ngb.best_val_loss_itr) + 1) if ngb.best_val_loss_itr is not None else 80
    print("NGBoost", iterations, flush=True)
    xgblss = XGBoostLSS(Gaussian(stabilization="None", response_fn="softplus", loss_fn="nll", initialize=False))
    dtrain, dvalid = xgb.DMatrix(x, label=y), xgb.DMatrix(vx, label=vy)
    xgblss.train(dict(eta=.05, max_depth=3, min_child_weight=30, seed=SEED, nthread=4,
        tree_method="hist", subsample=1., colsample_bytree=1.), dtrain, num_boost_round=80,
        evals=[(dtrain, "train"), (dvalid, "validation")], early_stopping_rounds=12, verbose_eval=False)
    xgb_iterations = int(xgblss.booster.best_iteration) + 1
    xgblss.booster = xgblss.booster[:xgb_iterations]
    print("XGBoostLSS", xgb_iterations, flush=True)
    base = pd.concat([train, valid], ignore_index=True)
    _, embeddings = tabm_predict(tabm, prep.transform(base[features]))
    bank = embeddings[base.event.to_numpy() == 1][-1000:]
    target = Path(target); target.mkdir(parents=True, exist_ok=True)
    import torch
    torch.save({"state_dict": tabm.state_dict(), "features": features, "k": 8, "width": 64,
                "blocks": 2, "dropout": .1, "seed": SEED}, target / "tabm.pt")
    np.save(target / "event_bank.npy", bank)
    state = {"preprocess": prep, "ngboost": ngb, "ngb_iterations": iterations,
        "xgboostlss": xgblss, "features": features}
    with (target / "trees_and_preprocess.pkl").open("wb") as stream:
        pickle.dump(state, stream)
    return {**state, "tabm": tabm, "bank": bank}, {"tabm": audit, "ngb_iterations": iterations,
        "xgb_iterations": xgb_iterations, "bank_events": len(bank)}


def raw_scores(state, data):
    import xgboost as xgb
    x = state["preprocess"].transform(data[state["features"]])
    p, hidden = tabm_predict(state["tabm"], x)
    ngb = state["ngboost"].pred_dist(x, max_iter=state["ngb_iterations"])
    xgb_pars = state["xgboostlss"].predict(xgb.DMatrix(x), pred_type="parameters")
    out = pd.DataFrame({"date": data.date.to_numpy(), "asset": data.asset.to_numpy(),
                        "tabm": p, "kl": kl_scores(hidden, state["bank"])})
    for name, mu, sigma in (("ngb", np.asarray(ngb.loc), np.asarray(ngb.scale)),
                             ("xgb", xgb_pars["loc"].to_numpy(), xgb_pars["scale"].to_numpy())):
        if not np.isfinite(mu).all() or not np.isfinite(sigma).all() or (sigma <= 0).any():
            raise ValueError("Модель вернула некорректное распределение")
        out[name + "_wide"] = norm.cdf(-1.5, mu, sigma) + norm.sf(1.5, mu, sigma)
        out[name + "_tail"] = norm.cdf(-2.5, mu, sigma) + norm.sf(2.5, mu, sigma)
    for column in ("kl", "ngb_wide", "xgb_wide"):
        out[column + "_z"] = out.groupby("date")[column].transform(
            lambda s: (s - s.mean()) / s.std(ddof=0) if s.std(ddof=0) > 1e-10 else s * 0)
    out["tabm_kl"] = out.kl_z
    out["tabm_ngboost_z"] = out.kl_z + out.ngb_wide_z
    out["tabm_both_z"] = out.kl_z + out.ngb_wide_z + out.xgb_wide_z
    return out


def offset_fit(p, y):
    offset = logit(np.clip(p, 1e-6, 1 - 1e-6))
    if len(y) < 30 or np.sum(y) < 3:
        return 0.
    fit = minimize_scalar(lambda b: np.mean(np.logaddexp(0, offset + b) - y * (offset + b)) + .0005 * b * b,
                          bounds=(-5, 5), method="bounded")
    return float(fit.x)


def calibrate(raw_cal, cal, raw_test, *, saved=None):
    y = cal.event.to_numpy(float)
    z = cal.target_z.abs().to_numpy()
    out, states = {}, {}
    for name in ("tabm", "tabm_kl", "tabm_ngboost_z", "tabm_both_z"):
        score = raw_cal[name].to_numpy()
        target = raw_test[name].to_numpy()
        if name == "tabm":
            score, target = logit(np.clip(score, 1e-6, 1 - 1e-6)), logit(np.clip(target, 1e-6, 1 - 1e-6))
        fit = minimize(lambda b: np.mean(np.logaddexp(0, b[0] + b[1] * score) - y * (b[0] + b[1] * score))
                       + .001 * b[1] ** 2, [logit(np.clip(y.mean(), .001, .999)), 0.],
                       method="L-BFGS-B", bounds=[(-12, 5), (0, 5)])
        if not fit.success:
            raise RuntimeError("Ошибка калибровки " + name)
        out[name] = expit(fit.x[0] + fit.x[1] * target)
        states[name] = fit.x.tolist()
    for prefix, name in (("ngb", "ngboost_c3"), ("xgb", "xgboostlss_c3")):
        wide = raw_cal[prefix + "_wide"].to_numpy()
        ratio = np.clip(raw_cal[prefix + "_tail"].to_numpy() / np.maximum(wide, 1e-10), 1e-6, 1 - 1e-6)
        bc, bq = offset_fit(wide, (z >= 1.5).astype(float)), offset_fit(ratio[z >= 1.5], y[z >= 1.5])
        twide = raw_test[prefix + "_wide"].to_numpy()
        tratio = raw_test[prefix + "_tail"].to_numpy() / np.maximum(twide, 1e-10)
        out[name] = expit(logit(np.clip(twide, 1e-6, 1 - 1e-6)) + bc) * expit(logit(np.clip(tratio, 1e-6, 1 - 1e-6)) + bq)
        states[name] = [bc, bq]
    if saved is not None:
        from .io import write_json as _write_json
        _write_json(saved, states)
    return pd.DataFrame(out, index=raw_test.index), states
