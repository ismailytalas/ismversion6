# -*- coding: utf-8 -*-
"""Altılı backtest: AGF vs model vs hücre bazlı model+AGF karışımı.

    python scripts/altili_backtest.py

Girdi : data/derived/oos_preds.parquet (scripts/walkforward_oos.py)
Çıktı : data/derived/blend_weights.csv, data/derived/altili_backtest.csv + konsol özeti

Adımlar
  1) Olasılık kaynakları (hepsi koşu içinde 1'e normalize):
       agf    : AGF yüzdesi
       model  : piyasasız walk-forward model
       blend_g: p ∝ pm^a · pa^b   (tek a,b — global)
       blend_h: aynı formül, (AGF entropi beşlisi × doğu) hücresine göre ayrı a,b
     a,b FIT döneminde maksimum olabilirlikle bulunur, TEST döneminde değerlendirilir.
  2) Altılı simülasyonu: her şehir-günün son 6 koşusu, greedy kupon (bütçe = kombinasyon).
  3) İkramiye vekili: kazanan kombinasyon başına ≈ (1-kesinti) / Π AGF_kazanan.
     'Halka karşı çarpan' = Σ_isabet (1/ΠAGF) / Σ maliyet. 1.0 = halkla aynı;
     kâr için > 1/(1-kesinti) gerekir. Gerçek ikramiyeler gelince bu vekil kalibre edilecek.
"""
from __future__ import annotations

import math
import pathlib

import numpy as np
import pandas as pd
from scipy.optimize import minimize

ROOT = pathlib.Path(__file__).resolve().parents[1]
D = ROOT / "data" / "derived"
K = ["tarih", "sehir", "kosu_no"]
DOGU = {"Şanlıurfa", "Diyarbakır", "Elazığ"}
FIT_SON = "2025-01-01"          # fit < bu tarih <= test
BUTCELER = [100, 500, 2000, 10000]
AGF_MIN = 0.005                  # yuvarlanmış %0 AGF'ler için taban


def hazirla() -> pd.DataFrame:
    d = pd.read_parquet(D / "oos_preds.parquet")
    d = d[d.groupby(K)["agf"].transform(lambda s: s.notna().all())].copy()
    d = d[d.groupby(K)["kazandi"].transform("sum") >= 1]          # kazananı belli koşular
    d["pa"] = d["agf"].clip(lower=AGF_MIN)
    d["pa"] /= d.groupby(K)["pa"].transform("sum")
    d["pm"] = d["p_model"].clip(lower=1e-4)
    d["pm"] /= d.groupby(K)["pm"].transform("sum")
    d["H"] = d.groupby(K)["pa"].transform(lambda p: -(p * np.log(p)).sum())
    d["dogu"] = d["sehir"].isin(DOGU).astype(int)
    d["fit"] = d["tarih"] < FIT_SON
    kenar = d[d.fit].drop_duplicates(K)["H"].quantile([.2, .4, .6, .8]).values
    d["Hq"] = np.digitize(d["H"], kenar) + 1
    d["hucre"] = "H" + d["Hq"].astype(str) + np.where(d["dogu"] == 1, "-doğu", "-diğer")
    d["rid"] = d.groupby(K, sort=False).ngroup()
    return d.reset_index(drop=True)


def _nll(w, lm, la, rid, y, n_r):
    s = w[0] * lm + w[1] * la
    m = np.full(n_r, -np.inf)
    np.maximum.at(m, rid, s)
    e = np.exp(s - m[rid])
    z = np.zeros(n_r)
    np.add.at(z, rid, e)
    lp = s - m[rid] - np.log(z[rid])
    return -(lp * y).sum() / y.sum()


def fit_ab(g: pd.DataFrame) -> tuple[float, float]:
    rid = g["rid"].factorize()[0]
    args = (np.log(g.pm.values), np.log(g.pa.values), rid, g.kazandi.values, rid.max() + 1)
    r = minimize(_nll, x0=[0.3, 0.9], args=args, method="Nelder-Mead",
                 options={"xatol": 1e-3, "fatol": 1e-6})
    return float(r.x[0]), float(r.x[1])


def uygula(d, a, b, mask, out):
    s = a * np.log(d.loc[mask, "pm"]) + b * np.log(d.loc[mask, "pa"])
    e = np.exp(s - s.groupby(d.loc[mask, "rid"]).transform("max"))
    d.loc[mask, out] = e / e.groupby(d.loc[mask, "rid"]).transform("sum")


def kupon(legs: list[np.ndarray], butce: float) -> list[int]:
    """legs: azalan sıralı olasılık dizileri. Greedy log-isabet/log-maliyet."""
    w = [1] * len(legs)
    cs = [np.cumsum(p) for p in legs]
    while True:
        best, bs = None, 0.0
        for i, c in enumerate(cs):
            n = w[i]
            if n >= len(c) or math.prod(w) / n * (n + 1) > butce:
                continue
            s = math.log(c[n] / c[n - 1]) / math.log((n + 1) / n)
            if s > bs:
                best, bs = i, s
        if best is None:
            return w
        w[best] += 1


def main():
    d = hazirla()
    print(f"{d.rid.nunique():,} koşu | fit < {FIT_SON} <= test")

    # --- 1) karışım ağırlıkları
    fitd = d[d.fit]
    ag, bg = fit_ab(fitd)
    uygula(d, ag, bg, d.index == d.index, "blend_g")
    satir = [{"hucre": "GLOBAL", "a_model": ag, "b_agf": bg, "kosu": fitd.rid.nunique()}]
    d["blend_h"] = np.nan
    for h, g in fitd.groupby("hucre"):
        a, b = fit_ab(g)
        satir.append({"hucre": h, "a_model": a, "b_agf": b, "kosu": g.rid.nunique()})
        uygula(d, a, b, d["hucre"] == h, "blend_h")
    W = pd.DataFrame(satir)
    W.to_csv(D / "blend_weights.csv", index=False)
    print("\nKarışım ağırlıkları  p ∝ model^a · AGF^b  (fit dönemi):")
    print(W.round(3).to_string(index=False))

    kaynaklar = ["pa", "pm", "blend_g", "blend_h"]
    ad = {"pa": "AGF", "pm": "Model", "blend_g": "Karışım-global", "blend_h": "Karışım-hücre"}
    te = d[~d.fit]
    kaz = te[te.kazandi == 1].drop_duplicates("rid")
    print("\nTEST — koşu başı log-loss (düşük iyi) ve favori isabeti:")
    for k in kaynaklar:
        top = te.loc[te.groupby("rid")[k].idxmax()]
        print(f"  {ad[k]:<15} logloss {-np.log(kaz[k]).mean():.4f}   top1 %{100*top.kazandi.mean():.1f}")
    print("  hücre bazında logloss (AGF → karışım-hücre):")
    for h, g in kaz.groupby("hucre"):
        print(f"    {h:<12} n={len(g):>5}  {-np.log(g.pa).mean():.4f} → {-np.log(g.blend_h).mean():.4f}")

    # --- 2) altılı simülasyonu
    races = {}
    for rid, g in d.groupby("rid", sort=False):
        races[rid] = g
    gunler = d.drop_duplicates("rid").sort_values("kosu_no").groupby(["tarih", "sehir"])["rid"].apply(list)
    gunler = gunler[gunler.map(len) >= 6]
    sonuc = []
    for (t, s), rids in gunler.items():
        legs = [races[r] for r in rids[-6:]]
        kazanan_agf = math.prod(float(g.loc[g.kazandi == 1, "pa"].iloc[0]) for g in legs)
        for k in kaynaklar:
            sirali = [g.sort_values(k, ascending=False) for g in legs]
            arr = [x[k].values for x in sirali]
            for B in BUTCELER:
                w = kupon(arr, B)
                isabet = all(x.kazandi.values[:n].any() for x, n in zip(sirali, w))
                sonuc.append({"tarih": t, "sehir": s, "kaynak": ad[k], "butce": B,
                              "komb": math.prod(w), "p_tahmin": math.prod(a[:n].sum() for a, n in zip(arr, w)),
                              "isabet": isabet, "vekil_odeme": (1 / kazanan_agf) if isabet else 0.0})
    R = pd.DataFrame(sonuc)
    R["test"] = R["tarih"] >= FIT_SON
    R["dogu"] = R["sehir"].isin(DOGU)
    R.to_csv(D / "altili_backtest.csv", index=False)

    def ozet(x):
        return pd.Series({"gun": len(x), "tahmin_%": 100 * x.p_tahmin.mean(), "isabet_%": 100 * x.isabet.mean(),
                          "ort_komb": x.komb.mean(),
                          "halka_karsi_carpan": x.vekil_odeme.sum() / x.komb.sum()})
    T = R[R.test]
    print("\nTEST — altılı (her şehir-günün son 6 koşusu):")
    print(T.groupby(["butce", "kaynak"]).apply(ozet).round(3).to_string())
    print("\nTEST — sadece doğu (Urfa/Diyarbakır/Elazığ):")
    print(T[T.dogu].groupby(["butce", "kaynak"]).apply(ozet).round(3).to_string())

    # çarpanın güven aralığı (ağır kuyruk) — bütçe 2000
    rng = np.random.default_rng(0)
    print("\nHalka karşı çarpan, %90 bootstrap aralığı (bütçe 2000, test):")
    for k, x in T[T.butce == 2000].groupby("kaynak"):
        v, c = x.vekil_odeme.values, x.komb.values
        bs = [v[i].sum() / c[i].sum() for i in (rng.integers(0, len(v), len(v)) for _ in range(1000))]
        print(f"  {k:<15} {v.sum()/c.sum():.3f}  [{np.quantile(bs,.05):.3f} – {np.quantile(bs,.95):.3f}]")


if __name__ == "__main__":
    main()
