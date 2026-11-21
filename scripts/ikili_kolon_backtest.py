# -*- coding: utf-8 -*-
"""Geniş İkili tek kolon backtest'i (gerçek TJK ödemeleriyle).

    python scripts/ikili_kolon_backtest.py

Soru: N atlık İkili kolon (N·(N-1)/2 kombinasyon) hangi koşularda kârlı?
Kolon sıralamaları:
  model      : walk-forward model olasılığına göre ilk N
  agf        : AGF'ye göre ilk N
  karisim    : p ∝ model^0.3 · AGF^0.7 ile ilk N
  favorisiz  : karışım sırasıyla ilk N, ama AGF favorisi kolona alınmaz
İsabet: koşunun ilk iki atı kolonun içinde. Getiri: İkili ödemesi (1 TL birim).
Kırılımlar: kaos seviyesi (surpriz_backtest.py ile aynı model), alan büyüklüğü, dönem.
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from surpriz_backtest import kosular, odemeler, kaos_ozellik, kaos_seviye, KAOS_X, FIT_SON, K  # noqa: E402

D = ROOT / "data" / "derived"
NLER = [4, 5, 6, 7, 8, 9, 10, 11, 12]


def kolon_siralari(d: pd.DataFrame) -> pd.DataFrame:
    """Her koşu için ilk iki atın her sıralamadaki yeri -> 'en_kotu' = kolonda olması için gereken N."""
    d = d.copy()
    s = 0.3 * np.log(d.pm.clip(1e-4)) + 0.7 * np.log(d.pa)
    d["pk"] = np.exp(s)
    d["pk"] /= d.groupby(K).pk.transform("sum")
    satir = []
    for key, g in d.groupby(K, sort=False):
        ilk2 = g.bitis <= 2
        if ilk2.sum() != 2:
            continue
        r = {"alan": len(g)}
        for ad, col in (("model", "pm"), ("agf", "pa"), ("karisim", "pk")):
            rank = g[col].rank(ascending=False, method="first")
            r[ad] = int(rank[ilk2].max())
        fav = g.pa.idxmax()
        if ilk2[fav]:
            r["favorisiz"] = 99                      # favori ilk ikide -> favorisiz kolon kaçırır
        else:
            rank = g.drop(index=fav).pk.rank(ascending=False, method="first")
            r["favorisiz"] = int(rank[ilk2.drop(index=fav)].max())
        satir.append((*key, *r.values()))
    return pd.DataFrame(satir, columns=K + ["alan", "model", "agf", "karisim", "favorisiz"])


def main():
    pd.set_option("display.width", 240)
    d = kosular()
    k = kaos_ozellik(d)
    fit = k.tarih < FIT_SON
    lr = LogisticRegression(max_iter=1000).fit(k.loc[fit, KAOS_X], k.loc[fit, "fav_kaybetti"])
    k["seviye"] = kaos_seviye(lr.predict_proba(k[KAOS_X])[:, 1])

    R = kolon_siralari(d).merge(k[K + ["seviye", "dogu"]], on=K).merge(odemeler()[K + ["ikili"]], on=K)
    R = R[R.ikili.notna()]
    R["donem"] = np.where(R.tarih < FIT_SON, "2022-24", "2025-26")
    R["alan_grup"] = pd.cut(R.alan, [0, 8, 11, 30], labels=["≤8", "9-11", "12+"])

    uzun = []
    for strat in ("model", "agf", "karisim", "favorisiz"):
        for n in NLER:
            x = R[R.alan > n].copy()                 # kolon tüm alanı kapsamasın
            x["strateji"], x["N"] = strat, n
            x["maliyet"] = n * (n - 1) / 2
            x["isabet"] = x[strat] <= n
            x["getiri"] = np.where(x.isabet, x.ikili, 0.0)
            uzun.append(x)
    U = pd.concat(uzun, ignore_index=True)
    U.to_csv(D / "ikili_kolon.csv", index=False)

    def roi(x):
        return 100 * (x.getiri.sum() / x.maliyet.sum() - 1)

    print("İsabette ortalama kazanç = isabetli ikili ödemesi / kolon maliyeti\n")
    print("=== ROI % — strateji × N, kaos seviyesine göre (tüm dönem) ===")
    print(U.groupby(["strateji", "N", "seviye"]).apply(roi).unstack().round(1).to_string())

    print("\n=== KAOTİK koşular: dönem tutarlılığı (ROI %, isabet %) ===")
    kk = U[U.seviye == "3-Kaotik"]
    t = kk.groupby(["strateji", "N", "donem"]).apply(
        lambda x: pd.Series({"kosu": len(x), "isabet_%": 100 * x.isabet.mean(), "ROI_%": roi(x),
                             "isabette_kat": (x.getiri[x.isabet] / x.maliyet[x.isabet]).median()}))
    print(t.round(1).unstack().to_string())

    print("\n=== KAOTİK, alan büyüklüğüne göre ROI % (tüm dönem) ===")
    print(kk.groupby(["strateji", "N", "alan_grup"]).apply(roi).unstack().round(1).to_string())

    rng = np.random.default_rng(0)
    print("\n=== En iyi adaylar — %90 bootstrap aralığı (kaotik, tüm dönem) ===")
    for (s, n), x in kk.groupby(["strateji", "N"]):
        g, c = x.getiri.values, x.maliyet.values
        bs = [100 * (g[i].sum() / c[i].sum() - 1) for i in (rng.integers(0, len(g), len(g)) for _ in range(500))]
        top = np.sort(g)[::-1][:5].sum() / max(g.sum(), 1)
        if roi(x) > -15:
            print(f"  {s:<10} N={n:<2} ROI {roi(x):6.1f}%  [{np.quantile(bs,.05):6.1f}, {np.quantile(bs,.95):6.1f}]"
                  f"  koşu {len(x)}  isabet {int(x.isabet.sum())}  en büyük 5 isabet payı {top:.2f}")


if __name__ == "__main__":
    main()
