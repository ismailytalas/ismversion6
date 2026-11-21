# -*- coding: utf-8 -*-
"""Değerli İkili seçimi: her koşuda sadece beklenen değeri yüksek çiftleri oyna.

    python scripts/ikili_deger_backtest.py

Çift olasılığı (Harville):  P(i,j) = p_i·p_j/(1-p_i) + p_i·p_j/(1-p_j)
  - 'bizim'  olasılık: karışım  p ∝ model^0.3 · AGF^0.7   (veya saf model)
  - 'piyasa' olasılığı: ganyan oranlarından (q_i ∝ 1/Gny)
Tahmini ikili ödemesi ≈ c / P_piyasa(i,j)  (c geçmiş veriden kalibre edilir, FIT dönemi)
Beklenen değer EV(i,j) = P_bizim(i,j) · c / P_piyasa(i,j)
Strateji: EV ≥ eşik olan tüm çiftleri 1'er TL oyna. Sonuç gerçek İkili ödemesiyle ölçülür.

Canlı kullanımda c/P_piyasa yerine vhs.tjk.org/muhtemeller'deki İkili muhtemel oranı
doğrudan kullanılabilir (tjk/muhtemeller.py) — bu tahminden daha isabetlidir.
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
ESIKLER = [1.0, 1.2, 1.5, 2.0, 3.0]


def harville_cift(p: np.ndarray) -> np.ndarray:
    pi, pj = p[:, None], p[None, :]
    m = pi * pj / (1 - pi) + pi * pj / (1 - pj)
    np.fill_diagonal(m, 0)
    return m


def main():
    pd.set_option("display.width", 220)
    d = kosular()
    d = d[d.gny.notna() & (d.gny > 0)]
    d["q"] = 100 / d.gny
    d["q"] /= d.groupby(K).q.transform("sum")
    d["pk"] = np.exp(0.3 * np.log(d.pm.clip(1e-4)) + 0.7 * np.log(d.pa))
    d["pk"] /= d.groupby(K).pk.transform("sum")

    k = kaos_ozellik(d)
    fit = k.tarih < FIT_SON
    lr = LogisticRegression(max_iter=1000).fit(k.loc[fit, KAOS_X], k.loc[fit, "fav_kaybetti"])
    k["seviye"] = kaos_seviye(lr.predict_proba(k[KAOS_X])[:, 1])
    pay = odemeler()[K + ["ikili"]].dropna()
    d = d.merge(pay, on=K)

    # c kalibrasyonu: gerçek ikili ödemesi × P_piyasa(kazanan çift)  (FIT dönemi medyanı)
    satirlar, kal = [], []
    for key, g in d.groupby(K, sort=False):
        bit = g.bitis.values
        if (bit <= 2).sum() != 2 or len(g) < 4:
            continue
        a, b = np.where(bit == 1)[0][0], np.where(bit == 2)[0][0]
        Q = harville_cift(g.q.values)
        if key[0] < FIT_SON:
            kal.append(g.ikili.iloc[0] * Q[a, b])
        satirlar.append((key, g, a, b, Q))
    c = float(np.median(kal))
    print(f"Kalibrasyon: ikili ödemesi ≈ {c:.3f} / P_piyasa(çift)   (n={len(kal)})")

    out = []
    for key, g, a, b, Q in satirlar:
        ikili = g.ikili.iloc[0]
        iu = np.triu_indices(len(g), 1)
        q = Q[iu]
        for ad, col in (("karisim", "pk"), ("model", "pm")):
            P = harville_cift(g[col].values)[iu]
            ev = P * c / q
            kazanan = (iu[0] == min(a, b)) & (iu[1] == max(a, b))
            for e in ESIKLER:
                sec = ev >= e
                n = int(sec.sum())
                if n:
                    hit = bool((sec & kazanan).any())
                    out.append((*key, ad, e, n, ikili if hit else 0.0, hit))
    R = pd.DataFrame(out, columns=K + ["olasilik", "esik", "bahis", "getiri", "isabet"])
    R = R.merge(k[K + ["seviye", "dogu"]], on=K)
    R["donem"] = np.where(R.tarih < FIT_SON, "2022-24", "2025-26")
    R.to_csv(D / "ikili_deger.csv", index=False)

    def oz(x):
        return pd.Series({"kosu": len(x), "ort_bahis": x.bahis.mean(), "isabet_%": 100 * x.isabet.mean(),
                          "ROI_%": 100 * (x.getiri.sum() / x.bahis.sum() - 1)})
    print("\n=== Tüm koşular ===")
    print(R.groupby(["olasilik", "esik", "donem"]).apply(oz).round(1).unstack().to_string())
    print("\n=== Kaos seviyesine göre ROI % (tüm dönem) ===")
    print(R.groupby(["olasilik", "esik", "seviye"]).apply(lambda x: 100 * (x.getiri.sum() / x.bahis.sum() - 1))
          .unstack().round(1).to_string())
    rng = np.random.default_rng(0)
    print("\n=== Kaotik + Orta, dönem bazında, %90 aralık ===")
    for (o_, e, dn), x in R[R.seviye != "1-Sakin"].groupby(["olasilik", "esik", "donem"]):
        g_, b_ = x.getiri.values, x.bahis.values
        bs = [100 * (g_[i].sum() / b_[i].sum() - 1) for i in (rng.integers(0, len(g_), len(g_)) for _ in range(500))]
        print(f"  {o_:<8} EV≥{e:<4} {dn}  ROI {100*(g_.sum()/b_.sum()-1):6.1f}%  [{np.quantile(bs,.05):6.1f}, "
              f"{np.quantile(bs,.95):6.1f}]  koşu {len(x)}  isabet {int(x.isabet.sum())}")


if __name__ == "__main__":
    main()
