# -*- coding: utf-8 -*-
"""Walk-forward (zamanda ileri) out-of-sample model tahminleri.

    python scripts/walkforward_oos.py [--basla 2022-01-01] [--periyot Q]

Her dönem (varsayılan çeyrek) için model YALNIZCA o dönemden önceki verilerle
eğitilir ve o dönem tahmin edilir. Böylece geçmişin her koşusu için,
o gün gerçekten elde olabilecek bir piyasasız p_win elde edilir.

Model: uygulamadaki temel_win ile aynı (picks.PARAMS, 800 tur, piyasa kolonları yok).
Çıktı: data/derived/oos_preds.parquet
"""
import sys, time, pathlib, argparse
import numpy as np
import pandas as pd
import lightgbm as lgb

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tjk.features import build_features, add_label_from_results  # noqa
from tjk.history import add_history_features  # noqa
from tjk.model import FEATURE_COLS, RACE_KEY  # noqa
from tjk.picks import PIYASA_KOLONLARI, PARAMS, kosu_ici_olasilik  # noqa

OUT = ROOT / "data" / "derived"
KEEP = RACE_KEY + ["at", "bitis", "kazandi", "agf", "gny", "alan_sayisi",
                   "kosu_cinsi", "mesafe", "pist", "ilk_yaris_mi", "maiden_mi",
                   "handikap_mi", "arap_mi"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--basla", default="2022-01-01")
    ap.add_argument("--periyot", default="QS")
    a = ap.parse_args()
    t0 = time.time()
    files = sorted((ROOT / "data" / "raw").glob("results_*.parquet"))
    raw = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    if "yurtdisi" in raw:
        raw = raw[~raw["yurtdisi"].fillna(False)].reset_index(drop=True)
    raw = add_label_from_results(raw)
    df = build_features(raw)
    df["kazandi"] = raw["kazandi"].values
    df["bitis"] = raw["bitis"].values
    print(f"özellikler: {len(df):,} satır ({time.time()-t0:.0f}s)", flush=True)
    df = add_history_features(df)
    print(f"geçmiş özellikleri tamam ({time.time()-t0:.0f}s)", flush=True)

    feats = [c for c in FEATURE_COLS if c in df.columns and c not in PIYASA_KOLONLARI]
    df["_t"] = pd.to_datetime(df["tarih"])
    df = df[df["bitis"].notna()]
    sinirlar = list(pd.date_range(a.basla, df["_t"].max() + pd.offsets.Day(1), freq=a.periyot))
    sinirlar.append(df["_t"].max() + pd.offsets.Day(1))
    parcalar = []
    for bas, son in zip(sinirlar[:-1], sinirlar[1:]):
        tr = df[df["_t"] < bas].dropna(subset=["kazandi"])
        te = df[(df["_t"] >= bas) & (df["_t"] < son)].copy()
        if te.empty:
            continue
        m = lgb.train(PARAMS, lgb.Dataset(tr[feats], label=tr["kazandi"]), 800)
        te["s_win"] = m.predict(te[feats])
        te = kosu_ici_olasilik(te, "s_win", "p_model")
        parcalar.append(te[[c for c in KEEP if c in te.columns] + ["p_model"]])
        print(f"{bas.date()} - {son.date()}: eğitim {len(tr):,}, test {len(te):,} "
              f"({time.time()-t0:.0f}s)", flush=True)
    out = pd.concat(parcalar, ignore_index=True)
    OUT.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT / "oos_preds.parquet", index=False)
    print(f"yazıldı: {OUT/'oos_preds.parquet'} ({len(out):,} satır)")


if __name__ == "__main__":
    main()
