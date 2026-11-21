# -*- coding: utf-8 -*-
"""TJK sonuç sayfalarından TÜM bahis ödemelerini toplar (6'lı Ganyan dahil).

Kullanım:
    python scripts/collect_payouts.py --test 2026-10-12            # tek gün, ham HTML'i de kaydeder
    python scripts/collect_payouts.py 2022-01-01 2026-10-12        # tam aralık (kaldığı yerden devam eder)
    python scripts/collect_payouts.py --yeniden-ayristir           # sadece altili.parquet'i yeniden üret

Sayfa yapısı (2026-10 itibarıyla): her koşu için bir div.bahisSonucGridContainer,
içinde her bahis için div.bahisSonucCard > h4 > [span bahis adı, span kombinasyon, span tutar].
Bir bahis kartı, o bahsin BİTTİĞİ koşunun altında durur (ör. 2. 6'lı Ganyan -> son ayak).

Çıktı:
    data/payouts/odemeler_YYYY-MM.parquet  -> tarih, sehir, kosu_no, bahis, kombinasyon, tutar, devir, ham
    data/payouts/altili.parquet             -> 6'lı Ganyan: altili_sira, ilk/son ayak, kazananlar, ikramiye, devir
"""
from __future__ import annotations

import re
import sys
import time
import pathlib
import argparse
import datetime as dt

import pandas as pd
from bs4 import BeautifulSoup

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OUT = ROOT / "data" / "payouts"
DEBUG = ROOT / "data" / "payouts_debug"
KOSU_RE = re.compile(r"(\d{1,2})\s*\.\s*Ko[şs]u")


def tutar_coz(metin: str) -> tuple[float | None, bool]:
    """'10.491,50 ₺' -> (10491.5, False);  'Bilen çıkmamıştır, 872,430.88 ₺ devretmiştir.' -> (872430.88, True)."""
    devir = "devret" in metin.lower() or "çıkmamış" in metin.lower()
    m = re.search(r"\d[\d.,]*", metin)
    if not m:
        return None, devir
    s = m.group(0).rstrip(".,")
    if re.search(r",\d{1,2}$", s):            # TR biçimi: 1.234,56
        s = s.replace(".", "").replace(",", ".")
    elif re.search(r"\.\d{1,2}$", s):         # EN biçimi: 1,234.56 (devir metinleri)
        s = s.replace(",", "")
    else:                                      # ondalıksız: 13.543.627 / 13,543,627
        s = s.replace(".", "").replace(",", "")
    try:
        return float(s), devir
    except ValueError:
        return None, devir


def sayfa_ayristir(html: str) -> list[dict]:
    soup = BeautifulSoup(html, "lxml")
    out = []
    for gi, grid in enumerate(soup.select(".bahisSonucGridContainer"), 1):
        bas = grid.find_previous(string=KOSU_RE)
        m = KOSU_RE.search(bas) if bas else None
        kosu_no = int(m.group(1)) if m else gi
        for kart in grid.select(".bahisSonucCard"):
            sp = [s.get_text(" ", strip=True) for s in kart.select("h4 span")]
            if not sp:
                continue
            ham = " | ".join(sp)
            tutar, devir = tutar_coz(sp[-1]) if len(sp) >= 2 else (None, False)
            out.append({"kosu_no": kosu_no, "bahis": sp[0],
                        "kombinasyon": sp[1] if len(sp) >= 3 else None,
                        "tutar": tutar, "devir": devir, "ham": ham})
    return out


def altili_tablosu(o: pd.DataFrame) -> pd.DataFrame:
    a = o[o["bahis"].str.contains(r"6\s*'?\s*L[Iİı]\s*GANYAN", case=False, regex=True, na=False)].copy()
    if a.empty:
        return a
    a["altili_sira"] = pd.to_numeric(a["bahis"].str.extract(r"^(\d)\.")[0], errors="coerce").fillna(1).astype(int)
    a["son_ayak"] = a["kosu_no"]
    a["ilk_ayak"] = a["kosu_no"] - 5
    a["ikramiye"] = a["tutar"].where(~a["devir"])
    a["devir_tutari"] = a["tutar"].where(a["devir"])
    return a[["tarih", "sehir", "altili_sira", "ilk_ayak", "son_ayak", "kombinasyon",
              "ikramiye", "devir_tutari", "ham"]].reset_index(drop=True)


def gun_cek(gun: dt.date, sleep: float, debug: bool = False) -> list[dict]:
    from tjk.scraper import get_cities, is_domestic, get_results_html
    rows = []
    for sid, ad in get_cities(gun):
        if not is_domestic(ad):
            continue
        try:
            html = get_results_html(gun, sid, ad)
        except Exception as e:
            print(f"  {gun} {ad}: hata {e}")
            continue
        if debug:
            DEBUG.mkdir(parents=True, exist_ok=True)
            (DEBUG / f"{gun}_{sid}.html").write_text(html, encoding="utf-8")
        k = sayfa_ayristir(html)
        for x in k:
            x.update(tarih=str(gun), sehir=ad)
        rows += k
        print(f"  {gun} {ad}: {len(k)} ödeme")
        time.sleep(sleep)
    return rows


def yeniden_ayristir():
    fs = sorted(OUT.glob("odemeler_*.parquet"))
    if not fs:
        print("odemeler_*.parquet yok"); return
    o = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
    a = altili_tablosu(o)
    a.to_parquet(OUT / "altili.parquet", index=False)
    print(f"altili.parquet: {len(a)} altılı | ikramiyeli {a['ikramiye'].notna().sum()} | "
          f"devirli {a['devir_tutari'].notna().sum()} | tüm ödemeler {len(o):,} satır")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bas", nargs="?")
    ap.add_argument("son", nargs="?")
    ap.add_argument("--test", help="tek gün (YYYY-MM-DD), ham HTML kaydeder")
    ap.add_argument("--html", nargs="*", help="kayıtlı HTML dosyalarını ayrıştır (ağ yok)")
    ap.add_argument("--sleep", type=float, default=1.0)
    ap.add_argument("--yeniden-ayristir", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 200)
    if a.yeniden_ayristir:
        return yeniden_ayristir()
    if a.html is not None or a.test:
        if a.html:
            rows = []
            for f in a.html:
                k = sayfa_ayristir(pathlib.Path(f).read_text(encoding="utf-8"))
                for x in k:
                    x.update(tarih="?", sehir=pathlib.Path(f).stem)
                rows += k
        else:
            rows = gun_cek(dt.date.fromisoformat(a.test), a.sleep, debug=True)
        o = pd.DataFrame(rows)
        print(o.groupby("bahis").size().to_string())
        print(altili_tablosu(o).drop(columns="ham").to_string())
        return
    bas, son = dt.date.fromisoformat(a.bas), dt.date.fromisoformat(a.son)
    ay = bas.replace(day=1)
    while ay <= son:
        sonraki = (ay.replace(day=28) + dt.timedelta(days=4)).replace(day=1)
        yol = OUT / f"odemeler_{ay:%Y-%m}.parquet"
        if yol.exists():
            print(f"{ay:%Y-%m} var, atlanıyor"); ay = sonraki; continue
        rows, gun = [], max(ay, bas)
        while gun < sonraki and gun <= son:
            try:
                rows += gun_cek(gun, a.sleep)
            except KeyboardInterrupt:
                sys.exit("durduruldu — tamamlanan aylar korundu, aynı komutla devam edin")
            except Exception as e:
                print(f"{gun}: {e}")
            gun += dt.timedelta(days=1)
        if rows:
            ay_tamam = (sonraki - dt.timedelta(days=1)) <= son
            hedef = yol if ay_tamam else OUT / f"odemeler_{ay:%Y-%m}_kismi.parquet"
            pd.DataFrame(rows).to_parquet(hedef, index=False)
            print(f"-> {hedef.name}: {len(rows):,} satır")
        ay = sonraki
    yeniden_ayristir()


if __name__ == "__main__":
    main()
