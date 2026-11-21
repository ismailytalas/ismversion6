# -*- coding: utf-8 -*-
"""Gerçek 6'lı Ganyan ikramiyeleriyle altılı backtest.

    python scripts/altili_gercek.py [--birim 1.0]

Girdi : data/derived/oos_preds.parquet   (walkforward_oos.py)
        data/payouts/altili.parquet       (collect_payouts.py)
Çıktı : data/derived/altili_gercek.csv + konsol özeti

1) Vekil kalibrasyonu: log(ikramiye) ~ log(1/Π AGF_kazanan). Eğim 1 ve sabit
   = log((1-kesinti)·birim) ise AGF-çarpımı vekili doğru demektir.
2) ROI: her altılı (1. ve 2. ayrı) için kaynak × bütçe kuponu kurulur.
   İsabette getiri = ikramiye × kapsanan kazanan kombinasyon sayısı (dead heat),
   maliyet = kombinasyon × birim. Devirli (bilen çıkmayan) altılılarda isabet
   ayrıca sayılır, getiriye katılmaz (o gün tek bilen biz olurduk; tutar bilinmiyor).
"""
from __future__ import annotations

import sys
import math
import pathlib
import argparse

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from altili_backtest import hazirla, fit_ab, uygula, kupon, DOGU, FIT_SON  # noqa: E402

D = ROOT / "data" / "derived"
BUTCELER = [100, 500, 2000, 10000]


def olasiliklar() -> pd.DataFrame:
    d = hazirla()
    fitd = d[d.fit]
    ag, bg = fit_ab(fitd)
    uygula(d, ag, bg, d.index == d.index, "blend_g")
    d["blend_h"] = np.nan
    for h, g in fitd.groupby("hucre"):
        a, b = fit_ab(g)
        uygula(d, a, b, d["hucre"] == h, "blend_h")
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--birim", type=float, default=1.0, help="altılı birim (misli) fiyatı, TL")
    a = ap.parse_args()

    alt = pd.read_parquet(ROOT / "data" / "payouts" / "altili.parquet")
    d = olasiliklar()
    races = {k: g for k, g in d.groupby(["tarih", "sehir", "kosu_no"], sort=False)}
    kaynaklar = {"pa": "AGF", "pm": "Model", "blend_g": "Karışım-global", "blend_h": "Karışım-hücre"}

    satir, eslesmeyen = [], 0
    for r in alt.itertuples():
        legs = [races.get((r.tarih, r.sehir, k)) for k in range(int(r.ilk_ayak), int(r.son_ayak) + 1)]
        if any(g is None for g in legs) or any(g.kazandi.sum() < 1 for g in legs):
            eslesmeyen += 1
            continue
        agf_kaz = math.prod(float(g.loc[g.kazandi == 1, "pa"].max()) for g in legs)
        base = {"tarih": r.tarih, "sehir": r.sehir, "altili_sira": r.altili_sira,
                "ikramiye": r.ikramiye, "devir": pd.notna(r.devir_tutari), "vekil": 1 / agf_kaz}
        for k, ad in kaynaklar.items():
            sirali = [g.sort_values(k, ascending=False) for g in legs]
            arr = [x[k].values for x in sirali]
            for B in BUTCELER:
                w = kupon(arr, B)
                kaz = [int(x.kazandi.values[:n].sum()) for x, n in zip(sirali, w)]
                n_kaz = math.prod(kaz)
                satir.append({**base, "kaynak": ad, "butce": B, "komb": math.prod(w),
                              "p_tahmin": math.prod(x[:n].sum() for x, n in zip(arr, w)),
                              "isabet": n_kaz > 0, "n_kazanan_komb": n_kaz})
    R = pd.DataFrame(satir)
    print(f"{alt.shape[0]} altılı, {eslesmeyen} eşleşmedi (şehir adı / koşu no / tarih aralığı)")
    if R.empty:
        return
    R["getiri"] = np.where(R.isabet & ~R.devir, R.ikramiye.fillna(0) * R.n_kazanan_komb, 0.0)
    R["maliyet"] = R.komb * a.birim
    R["test"] = R.tarih >= FIT_SON
    R["dogu"] = R.sehir.isin(DOGU)
    R.to_csv(D / "altili_gercek.csv", index=False)

    # 1) vekil kalibrasyonu
    v = R.drop_duplicates(["tarih", "sehir", "altili_sira"])
    v = v[v.ikramiye.notna() & (v.ikramiye > 0)]
    x, y = np.log(v.vekil), np.log(v.ikramiye)
    egim, sabit = np.polyfit(x, y, 1)
    print(f"\nVekil kalibrasyonu (n={len(v)}): log(ikramiye) = {sabit:.2f} + {egim:.3f}·log(1/ΠAGF)"
          f"   korelasyon {np.corrcoef(x, y)[0,1]:.3f}")
    print(f"  eğim=1 varsayımıyla ölçek: ikramiye ≈ {np.exp((y - x).median()):.3f} / ΠAGF  (medyan)")
    print(f"  devirli altılı oranı: %{100*R.drop_duplicates(['tarih','sehir','altili_sira']).devir.mean():.1f}")

    # 2) ROI
    def ozet(g):
        return pd.Series({"altili": len(g), "isabet_%": 100 * g.isabet.mean(),
                          "devirde_isabet": int((g.isabet & g.devir).sum()),
                          "maliyet": g.maliyet.sum(), "getiri": g.getiri.sum(),
                          "ROI_%": 100 * (g.getiri.sum() / g.maliyet.sum() - 1)})
    for ad, m in [("TÜM DÖNEM", R.index == R.index), ("TEST (2025+)", R.test), ("DOĞU, tüm dönem", R.dogu)]:
        print(f"\n{ad}:")
        print(R[m].groupby(["butce", "kaynak"]).apply(ozet).round(1).to_string())

    rng = np.random.default_rng(0)
    print("\nROI %90 bootstrap aralığı (tüm dönem):")
    for (B, k), g in R.groupby(["butce", "kaynak"]):
        gt, mt = g.getiri.values, g.maliyet.values
        bs = [100 * (gt[i].sum() / mt[i].sum() - 1) for i in (rng.integers(0, len(gt), len(gt)) for _ in range(1000))]
        print(f"  {B:>6} {k:<15} {100*(gt.sum()/mt.sum()-1):7.1f}%  [{np.quantile(bs,.05):7.1f} , {np.quantile(bs,.95):7.1f}]")


if __name__ == "__main__":
    main()
