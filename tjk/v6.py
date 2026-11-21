# -*- coding: utf-8 -*-
"""tjk-v6 hesap katmanı: kaos skoru, model+AGF karışımı, İkili önerileri, Altılı yardımcıları.

Tüm parametreler models/v6_config.json'dan okunur (backtest scriptlerinin çıktısı):
  scripts/surpriz_backtest.py  -> kaos katsayıları
  scripts/altili_backtest.py   -> karışım ağırlıkları (hücre = AGF entropi beşlisi × doğu)
  scripts/ikili_deger_backtest.py -> İkili ödeme kalibrasyonu c
"""
from __future__ import annotations

import json
import math
import pathlib
import unicodedata
from itertools import combinations

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
CFG = json.loads((ROOT / "models" / "v6_config.json").read_text(encoding="utf-8"))
RACE_KEY = ["tarih", "sehir", "kosu_no"]


def _fold(s) -> str:
    s = unicodedata.normalize("NFKD", str(s).replace("ı", "i").replace("İ", "I"))
    return "".join(c for c in s if not unicodedata.combining(c)).lower()


DOGU = {_fold(x) for x in CFG["dogu"]}


def dogu_mu(sehir) -> bool:
    f = _fold(sehir)
    return any(x in f for x in DOGU)


def birim_fiyat(tarih, sehir) -> float:
    """6'lı Ganyan birim fiyatı (TL)."""
    t = str(tarih)[:10]
    b = CFG["altili"]["birim_tablosu"][-1]
    for satir in CFG["altili"]["birim_tablosu"]:
        if t >= satir[0]:
            b = satir
    return b[2] if dogu_mu(sehir) else b[1]


# --------------------------------------------------------------------------- olasılıklar
def kosu_hazirla(rd: pd.DataFrame) -> pd.DataFrame:
    """Tek koşu: pa (AGF), pq (ganyan), pk (karışım), sıralar."""
    d = rd.dropna(subset=["p_win"]).copy()
    d["pm"] = d["p_win"] / d["p_win"].sum()
    # halk olasılığı: AGF, yoksa ganyan oranı, o da yoksa model
    agf = pd.to_numeric(d.get("agf"), errors="coerce")
    oran = pd.to_numeric(d.get("oran"), errors="coerce")
    if agf.notna().sum() >= 2:
        pa = agf.fillna(agf.min() / 2).clip(lower=CFG["agf_min"])
        d["agf_kaynak"] = "AGF"
    elif oran.notna().sum() >= 2:
        pa = (1 / oran).fillna((1 / oran).min() / 2)
        d["agf_kaynak"] = "Ganyan"
    else:
        pa = d["pm"].copy()
        d["agf_kaynak"] = "Model"
    d["pa"] = pa / pa.sum()
    pq = (1 / oran) if oran.notna().sum() >= 2 else d["pa"]
    pq = pq.fillna(pq.min() / 2 if pq.notna().any() else 0.01)
    d["pq"] = pq / pq.sum()

    H = float(-(d.pa * np.log(d.pa)).sum())
    kenar = CFG["karisim"]["H_esikleri"]
    hucre = f"H{int(np.digitize(H, kenar)) + 1}-{'doğu' if dogu_mu(d['sehir'].iloc[0]) else 'diğer'}"
    a, b = CFG["karisim"]["agirliklar"].get(hucre, CFG["karisim"]["agirliklar"]["GLOBAL"])
    s = a * np.log(d.pm.clip(1e-4)) + b * np.log(d.pa)
    e = np.exp(s - s.max())
    d["pk"] = e / e.sum()
    d["H"], d["hucre"] = H, hucre
    d["agf_sira"] = d.pa.rank(ascending=False, method="first").astype(int)
    d["model_sira"] = d.pm.rank(ascending=False, method="first").astype(int)
    d["karisim_sira"] = d.pk.rank(ascending=False, method="first").astype(int)
    return d.sort_values("pk", ascending=False)


def kaos(d: pd.DataFrame) -> dict:
    """P(AGF favorisi kazanamaz) ve seviye."""
    fav = d.loc[d.pa.idxmax()]
    x = {"H": d.H.iloc[0], "fav_pa": d.pa.max(), "log_alan": math.log(len(d)),
         "maiden": float(d.get("maiden_mi", pd.Series([0])).fillna(0).iloc[0]),
         "handikap": float(d.get("handikap_mi", pd.Series([0])).fillna(0).iloc[0]),
         "arap": float(d.get("arap_mi", pd.Series([0])).fillna(0).iloc[0]),
         "dogu": float(dogu_mu(d["sehir"].iloc[0])),
         "uyusmaz": float(fav["model_sira"] != 1)}
    k = CFG["kaos"]
    z = k["sabit"] + sum(c * x[f] for f, c in zip(k["ozellikler"], k["katsayi"]))
    p = 1 / (1 + math.exp(-z))
    e1, e2 = k["esikler"]
    seviye = "Sakin" if p < e1 else ("Orta" if p < e2 else "Kaotik")
    return {"p": p, "seviye": seviye, "ikon": {"Sakin": "🟢", "Orta": "🟡", "Kaotik": "🔴"}[seviye]}


# --------------------------------------------------------------------------- İkili
def harville(p: np.ndarray) -> np.ndarray:
    pi, pj = p[:, None], p[None, :]
    m = pi * pj / (1 - pi) + pi * pj / (1 - pj)
    np.fill_diagonal(m, 0)
    return m


def _no(x):
    return "?" if pd.isna(x) else str(int(x))


def ikili_cift_tablosu(d: pd.DataFrame) -> pd.DataFrame:
    """Tüm çiftler: karışım olasılığı ve ganyandan tahmini ödeme."""
    P = harville(d.pk.values)
    Q = harville(d.pq.values)
    c = CFG["ikili"]["odeme_kalibrasyonu_c"]
    no, at = d["no"].values, d["at"].values
    s = []
    for i, j in combinations(range(len(d)), 2):
        s.append({"i": i, "j": j, "Çift": f"{_no(no[i])}-{_no(no[j])}",
                  "Atlar": f"{at[i]} / {at[j]}", "p": P[i, j],
                  "odeme": c / max(Q[i, j], 1e-6)})
    return pd.DataFrame(s).sort_values("p", ascending=False).reset_index(drop=True)


def ikili_onerileri(d: pd.DataFrame, agf_min: int | None = None, model_max: int | None = None,
                    n_guvenli: int = 3) -> dict:
    """agf_min: sürprizin halk (AGF) sırası en az; model_max: modelde en çok kaçıncı (varsayılan: config)."""
    t = ikili_cift_tablosu(d)
    guvenli = t.head(n_guvenli)
    ic = CFG["ikili"]
    agf_min = ic["surpriz_agf_sira_min"] if agf_min is None else agf_min
    model_max = ic["surpriz_model_sira_max"] if model_max is None else model_max
    m1 = d[d.model_sira == 1].index[0]
    sur = d[(d.agf_sira >= agf_min) & (d.model_sira <= model_max) & (d.index != m1)]
    pos = {idx: k for k, idx in enumerate(d.index)}
    mp, sp = pos[m1], {pos[x] for x in sur.index}
    yuksek = t[((t.i == mp) & t.j.isin(sp)) | ((t.j == mp) & t.i.isin(sp))]
    return {"guvenli": guvenli, "yuksek": yuksek, "surpriz_atlar": sur,
            "model1": d.loc[m1]}


def kolon_hesapla(d: pd.DataFrame, nolar: list) -> dict | None:
    sec = d[d["no"].isin(nolar)]
    n = len(sec)
    if n < 2:
        return None
    t = ikili_cift_tablosu(d)
    idx = {k for k, v in enumerate(d["no"].values) if v in set(nolar)}
    kt = t[t.i.isin(idx) & t.j.isin(idx)]
    komb = n * (n - 1) // 2
    p = float(kt.p.sum())
    ev = float((kt.p * kt.odeme).sum() / komb)
    return {"komb": komb, "p": min(p, 1.0), "odeme_min": kt.odeme.min(), "odeme_med": kt.odeme.median(),
            "odeme_max": kt.odeme.max(), "kat_med": kt.odeme.median() / komb, "ev": ev, "tablo": kt}


# --------------------------------------------------------------------------- Altılı
def altili_oneri(ayaklar: list[pd.DataFrame], hedef_komb: int, max_at: int = 8) -> list[list]:
    """Greedy: log(isabet) artışı / log(maliyet) artışı en yüksek at eklenir (karışım olasılığıyla)."""
    p = [a.pk.values for a in ayaklar]
    w = [1] * len(ayaklar)
    cs = [np.cumsum(x) for x in p]
    while True:
        best, bs = None, 0.0
        for i, c in enumerate(cs):
            n = w[i]
            if n >= min(len(c), max_at) or math.prod(w) / n * (n + 1) > hedef_komb:
                continue
            s = math.log(c[n] / c[n - 1]) / math.log((n + 1) / n)
            if s > bs:
                best, bs = i, s
        if best is None:
            break
        w[best] += 1
    return [a["no"].values[:n].tolist() for a, n in zip(ayaklar, w)]


def altili_ozet(ayaklar: list[pd.DataFrame], secim: list[list], misli: float) -> dict:
    komb = math.prod(max(len(s), 0) for s in secim)
    p = math.prod(float(a[a["no"].isin(s)].pk.sum()) for a, s in zip(ayaklar, secim))
    h = math.prod(float(a[a["no"].isin(s)].pa.sum()) for a, s in zip(ayaklar, secim))
    birim = birim_fiyat(ayaklar[0]["tarih"].iloc[0], ayaklar[0]["sehir"].iloc[0])
    return {"komb": komb, "p": p, "halk": h, "birim": birim, "maliyet": komb * birim * misli,
            "avantaj": p / h if h > 0 else np.nan}


def tek_at_alternatifleri(ayaklar: list[pd.DataFrame], secim: list[list], ayak_no: list[int],
                          misli: float, n: int = 4) -> pd.DataFrame:
    """Tek at (banko) bırakılan ayaklarda sıradaki adaylar ve eklenirse kuponun yeni hali."""
    oz = altili_ozet(ayaklar, secim, misli)
    s = []
    for a, sec, k in zip(ayaklar, secim, ayak_no):
        if len(sec) != 1:
            continue
        banko = a[a["no"].isin(sec)].iloc[0]
        for r in a[~a["no"].isin(sec)].sort_values("pk", ascending=False).head(n).itertuples():
            s.append({"Koşu": k, "Banko": f"{_no(banko['no'])} {banko['at']} (%{100*banko['pk']:.0f})",
                      "Alternatif": f"{_no(r.no)} {r.at}", "Karışım %": round(100 * r.pk, 1),
                      "AGF sıra": int(r.agf_sira), "Model sıra": int(r.model_sira),
                      "Eklenirse isabet %": round(100 * oz["p"] / banko["pk"] * (banko["pk"] + r.pk), 2),
                      "Eklenirse bedel (TL)": round(oz["maliyet"] * 2, 2)})
    return pd.DataFrame(s)


def yedek_at_tablosu(ayaklar: list[pd.DataFrame], ayak_no: list[int], n: int = 6) -> pd.DataFrame:
    """Her ayakta ilk n at ve kümülatif kapsama: kaç at alınırsa ayağın yüzde kaçı kapanır."""
    s = []
    for a, k in zip(ayaklar, ayak_no):
        kum = 0.0
        for i, r in enumerate(a.sort_values("pk", ascending=False).head(n).itertuples(), 1):
            kum += r.pk
            s.append({"Koşu": k, "Sıra": i, "No": _no(r.no), "At": r.at, "Karışım %": round(100 * r.pk, 1),
                      "Kümülatif %": round(100 * kum, 1), "AGF sıra": int(r.agf_sira),
                      "Model sıra": int(r.model_sira)})
    return pd.DataFrame(s)
