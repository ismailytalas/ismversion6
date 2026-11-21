# -*- coding: utf-8 -*-
"""Kaos skoru + sürpriz Plase / İkili / Ganyan backtest'i (gerçek TJK ödemeleriyle).

    python scripts/surpriz_backtest.py

Girdi : data/derived/oos_preds.parquet  (walk-forward model olasılıkları)
        data/payouts/odemeler_*.parquet  (collect_payouts.py)
Çıktı : data/derived/kaos_model.json, data/derived/surpriz_bahisler.csv + konsol özeti

KAOS SKORU = P(AGF favorisi kazanamaz), koşu öncesi bilinen bilgilerden lojistik regresyon:
  AGF entropisi, favorinin AGF payı, alan büyüklüğü, maiden/handikap/arap, doğu hipodromu,
  model ile AGF favorisinin aynı olup olmadığı. FIT (<2025) döneminde eğitilir.
  Seviye: Sakin (<%55) / Orta (%55–70) / Kaotik (>%70)

ÖDEME EŞLEMESİ (program numarası olmadan):
  Sıralı İkili kartı 1. ve 2. atın numarasını verir -> Plase kartları 1. ve 2. ata atanır.
  İkili ödemesi {1., 2.} çiftine, Ganyan ödemesi 1. ata aittir. Dead-heat koşuları atlanır.
  Tüm ödemeler 1 TL'lik bahis için; her bahis 1 TL varsayılır.

NOT: AGF ve Gny sonuç sayfasındaki KAPANIŞ değerleridir. Gerçek bahis anında elde olan AGF
biraz farklıdır; AGF'ye dayanan stratejilerde sonuçlar hafif iyimser olabilir.
"""
from __future__ import annotations

import json
import pathlib
import itertools

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

ROOT = pathlib.Path(__file__).resolve().parents[1]
D = ROOT / "data" / "derived"
K = ["tarih", "sehir", "kosu_no"]
DOGU = {"Şanlıurfa", "Diyarbakır", "Elazığ"}
FIT_SON = "2025-01-01"
KAOS_ESIK = (0.55, 0.70)
AGF_MIN = 0.005


# --------------------------------------------------------------------------- veri
def odemeler() -> pd.DataFrame:
    o = pd.concat([pd.read_parquet(f) for f in sorted((ROOT / "data" / "payouts").glob("odemeler_*.parquet"))],
                  ignore_index=True)
    o = o[~o["devir"].fillna(False)]
    si = o[o.bahis == "SIRALI İKİLİ"].groupby(K).kombinasyon.first().str.split("/", expand=True)
    si = si.iloc[:, :2]
    si.columns = ["n1", "n2"]
    pl = o[o.bahis == "PLASE"].groupby(K).apply(lambda g: dict(zip(g.kombinasyon, g.tutar)))
    r = si.join(pl.rename("pl"), how="left")
    r["plase1"] = [p.get(a) if isinstance(p, dict) else np.nan for a, p in zip(r.n1, r.pl)]
    r["plase2"] = [p.get(b) if isinstance(p, dict) else np.nan for b, p in zip(r.n2, r.pl)]
    r["ikili"] = o[o.bahis == "İKİLİ"].groupby(K).tutar.first()
    r["ganyan"] = o[o.bahis == "GANYAN"].groupby(K).tutar.first()
    return r.drop(columns=["pl", "n1", "n2"]).reset_index()


def kosular() -> pd.DataFrame:
    d = pd.read_parquet(D / "oos_preds.parquet")
    ok = d.groupby(K).agf.transform(lambda s: s.notna().all())
    d = d[ok].copy()
    # dead-heat (ilk 2'de eşit derece) koşuları çıkar
    dh = d[d.bitis <= 2].groupby(K).bitis.transform(lambda s: s.duplicated(keep=False).any())
    kotu = d.loc[dh[dh].index, K].drop_duplicates()
    d = d.merge(kotu.assign(_x=1), on=K, how="left")
    d = d[d._x.isna()].drop(columns="_x")
    d["pa"] = d.agf.clip(lower=AGF_MIN)
    d["pa"] /= d.groupby(K).pa.transform("sum")
    d["pm"] = d.p_model / d.groupby(K).p_model.transform("sum")
    d["agf_sira"] = d.groupby(K).pa.rank(ascending=False, method="first")
    d["model_sira"] = d.groupby(K).pm.rank(ascending=False, method="first")
    d["deger"] = d.pm / d.pa
    return d


# --------------------------------------------------------------------------- kaos
KAOS_X = ["H", "fav_pa", "log_alan", "maiden", "handikap", "arap", "dogu", "uyusmaz"]


def kaos_ozellik(d: pd.DataFrame) -> pd.DataFrame:
    g = d.groupby(K)
    r = pd.DataFrame({
        "H": g.pa.apply(lambda p: float(-(p * np.log(p)).sum())),
        "fav_pa": g.pa.max(),
        "log_alan": np.log(g.size()),
        "maiden": g.maiden_mi.first().fillna(0),
        "handikap": g.handikap_mi.first().fillna(0),
        "arap": g.arap_mi.first().fillna(0),
        "uyusmaz": g.apply(lambda x: float(x.loc[x.pa.idxmax(), "model_sira"] != 1)),
        "fav_kaybetti": g.apply(lambda x: float(x.loc[x.pa.idxmax(), "bitis"] != 1)),
    }).reset_index()
    r["dogu"] = r.sehir.isin(DOGU).astype(float)
    return r


def kaos_seviye(p):
    return np.select([p < KAOS_ESIK[0], p < KAOS_ESIK[1]], ["1-Sakin", "2-Orta"], "3-Kaotik")


# --------------------------------------------------------------------------- stratejiler
def bahisler(d: pd.DataFrame, pay: pd.DataFrame) -> pd.DataFrame:
    """Her koşu için strateji bahislerini ve getirisini üretir (1 TL / bahis)."""
    d = d.merge(pay, on=K, how="inner")
    out = []
    for key, g in d.groupby(K, sort=False):
        g = g.set_index("at")
        bit = g.bitis
        ilk2 = set(bit[bit <= 2].index)
        ag = g.sort_values("pa", ascending=False).index.tolist()
        mo = g.sort_values("pm", ascending=False).index.tolist()
        sur = g[(g.agf_sira >= 6) & (g.model_sira <= 3)].index.tolist()          # sürpriz: halk 6+, model ilk 3
        deg = g[(g.deger >= 2) & (g.model_sira <= 5)].index.tolist()             # değerli: model/AGF ≥ 2
        r0 = g.iloc[0]

        def plase(at):
            b = bit[at]
            return r0.plase1 if b == 1 else (r0.plase2 if b == 2 else 0.0)

        def ikili(a, b):
            return r0.ikili if {a, b} == ilk2 else 0.0

        def ganyan(at):
            return r0.ganyan if bit[at] == 1 else 0.0

        kayit = []
        if pd.notna(r0.plase1) and pd.notna(r0.plase2):
            kayit += [("Plase", "AGF favorisi", [plase(ag[0])]),
                      ("Plase", "Model 1.", [plase(mo[0])]),
                      ("Plase", "Sürpriz (AGF 6+, model ilk 3)", [plase(a) for a in sur]),
                      ("Plase", "Değerli (model/AGF ≥ 2)", [plase(a) for a in deg])]
        if pd.notna(r0.ikili):
            kayit += [("İkili", "AGF ilk 2", [ikili(ag[0], ag[1])]),
                      ("İkili", "Model ilk 2", [ikili(mo[0], mo[1])]),
                      ("İkili", "Model 1. × sürprizler", [ikili(mo[0], a) for a in sur if a != mo[0]]),
                      ("İkili", "AGF fav × sürprizler", [ikili(ag[0], a) for a in sur if a != ag[0]]),
                      ("İkili", "Model ilk 5 kolon (10)", [ikili(a, b) for a, b in itertools.combinations(mo[:5], 2)]),
                      ("İkili", "AGF ilk 5 kolon (10)", [ikili(a, b) for a, b in itertools.combinations(ag[:5], 2)])]
        if pd.notna(r0.ganyan):
            kayit += [("Ganyan", "AGF favorisi", [ganyan(ag[0])]),
                      ("Ganyan", "Model 1.", [ganyan(mo[0])]),
                      ("Ganyan", "Sürpriz (AGF 6+, model ilk 3)", [ganyan(a) for a in sur])]
        for tur, ad, getiriler in kayit:
            if getiriler:
                out.append((*key, tur, ad, len(getiriler), float(np.nansum(getiriler)),
                            int(sum(x > 0 for x in getiriler))))
    return pd.DataFrame(out, columns=K + ["tur", "strateji", "bahis", "getiri", "isabet"])


def ozet(x: pd.DataFrame, rng, n_boot=500) -> pd.Series:
    b, g = x.bahis.values, x.getiri.values
    idx = (rng.integers(0, len(b), len(b)) for _ in range(n_boot))
    bs = [g[i].sum() / b[i].sum() - 1 for i in idx]
    return pd.Series({"kosu": len(x), "bahis": int(b.sum()), "isabet_%": 100 * x.isabet.sum() / b.sum(),
                      "ROI_%": 100 * (g.sum() / b.sum() - 1),
                      "alt_%": 100 * np.quantile(bs, .05), "ust_%": 100 * np.quantile(bs, .95)})


def main():
    pd.set_option("display.width", 220)
    rng = np.random.default_rng(0)
    d = kosular()

    # --- kaos modeli
    k = kaos_ozellik(d)
    fit = k.tarih < FIT_SON
    lr = LogisticRegression(max_iter=1000).fit(k.loc[fit, KAOS_X], k.loc[fit, "fav_kaybetti"])
    k["kaos"] = lr.predict_proba(k[KAOS_X])[:, 1]
    k["seviye"] = kaos_seviye(k.kaos)
    json.dump({"ozellikler": KAOS_X, "katsayi": lr.coef_[0].tolist(), "sabit": float(lr.intercept_[0]),
               "esikler": KAOS_ESIK}, open(D / "kaos_model.json", "w"), ensure_ascii=False, indent=1)
    print("KAOS SKORU = P(AGF favorisi kazanamaz) — katsayılar:")
    for c, v in zip(KAOS_X, lr.coef_[0]):
        print(f"  {c:<10} {v:+.3f}")
    te = k[~fit]
    print("\nTEST dönemi kalibrasyonu (tahmin vs gerçek favori kaybı):")
    t = te.groupby("seviye").agg(kosu=("kaos", "size"), tahmin=("kaos", "mean"), gercek=("fav_kaybetti", "mean"))
    print((t.assign(tahmin=lambda x: 100 * x.tahmin, gercek=lambda x: 100 * x.gercek)).round(1).to_string())

    # --- bahis backtest
    pay = odemeler()
    B = bahisler(d, pay).merge(k[K + ["kaos", "seviye", "dogu"]], on=K)
    B["donem"] = np.where(B.tarih < FIT_SON, "2022-24", "2025-26")
    B.to_csv(D / "surpriz_bahisler.csv", index=False)

    print("\n=== TÜM KOŞULAR (tüm dönem) ===")
    print(B.groupby(["tur", "strateji"]).apply(ozet, rng).round(1).to_string())
    print("\n=== KAOS SEVİYESİNE GÖRE ROI % (tüm dönem) ===")
    print(B.groupby(["tur", "strateji", "seviye"]).apply(lambda x: 100 * (x.getiri.sum() / x.bahis.sum() - 1))
          .unstack().round(1).to_string())
    print("\n=== KAOTİK koşular — dönem bazında (2022-24 / 2025-26 tutarlılık testi) ===")
    print(B[B.seviye == "3-Kaotik"].groupby(["tur", "strateji", "donem"]).apply(ozet, rng)
          .round(1).to_string())
    print("\n=== KAOTİK + DOĞU ===")
    print(B[(B.seviye == "3-Kaotik") & (B.dogu == 1)].groupby(["tur", "strateji"]).apply(ozet, rng)
          .round(1).to_string())


if __name__ == "__main__":
    main()
