# -*- coding: utf-8 -*-
"""TJK v6 — Kaos Radarı, İkili önerileri ve Altılı seçim aracı.

    streamlit run tjk_v6_app.py
"""
import datetime as dt

import pandas as pd
import streamlit as st

from tjk import scraper, sonuclar, eda, v6
from tjk.picks import (gecmisi_yukle, gunun_programi, ozellik_uret,
                       modelleri_hazirla, tahmin_et, surpriz_notlari, PIYASA_KOLONLARI)
from tjk.model import FEATURE_COLS



def genis(eleman, *args, **kw):
    """Tablo/grafiği tam genişlikte çizer. Yeni Streamlit width="stretch", eskisi use_container_width ister;
    sürüm numarasına güvenmek yerine önce yenisini dener, olmazsa eskisine düşer."""
    try:
        return eleman(*args, width="stretch", **kw)
    except Exception as e:  # TypeError / StreamlitAPIException (eski sürüm)
        if "stretch" not in str(e) and not isinstance(e, TypeError):
            raise
        return eleman(*args, use_container_width=True, **kw)

st.set_page_config(page_title="TJK v6 — Kaos Radarı", page_icon="🐎", layout="wide")
st.title("🐎 TJK v6 — Kaos Radarı")
st.caption("Koşunun kaotik olup olmadığını, ikili önerilerini ve altılı kuponunu gösterir. "
           "Neyi oynayacağına sen karar verirsin. Bu bir tahmin aracıdır, kazanç garantisi yoktur.")

ss = st.session_state
ss.setdefault("yenile", 0)

# --- Kenar çubuğu ------------------------------------------------------------------
with st.sidebar:
    st.header("Ayarlar")
    tarih = st.date_input("Tarih", dt.date.today(), format="DD/MM/YYYY")
    if st.button("🔄 Programı / AGF'yi yenile",
                 help="TJK'dan programı ve güncel AGF'yi yeniden çeker; tahminler yeniden üretilir."):
        ss.yenile += 1
    st.subheader("Sürpriz tanımı")
    agf_esigi = st.slider("Halkın AGF sırası en az", 3, 12, 6,
                          help="6 = halkın ilk 5 favorisi sürpriz sayılmaz. Geçmiş test 6 ile yapıldı.")
    model_esigi = st.slider("Model sıralamasında en çok", 1, 8, 3,
                            help="Model atı ilk kaçta görmeli. Geçmiş test 3 ile yapıldı.")
    if (agf_esigi, model_esigi) != (6, 3):
        st.caption("⚠️ Geçmiş test sonuçları (+%4,8 vb.) 6 / 3 ayarı içindir.")
    st.subheader("Altılı")
    misli = st.number_input("Misli", 1, 100, 1, 1)
    st.caption("Kupon bedeli = kombinasyon × birim fiyat × misli. Birim fiyat hipodroma göre "
               "otomatik (2026: büyük hipodromlar 1,25 TL, Urfa/Elazığ/Diyarbakır 1,00 TL).")
    st.subheader("İkili")
    ikili_misli = st.number_input("İkili misli", 1, 100, 1, 1)
    yeniden = st.checkbox("Modeli yeniden eğit", value=False,
                          help="Yeni geçmiş veri indirdiysen işaretle (birkaç dakika sürer).")
    with st.expander("Kaos seviyeleri ne demek?"):
        st.markdown(
            "**Kaos** = AGF favorisinin kazanamama olasılığı. Halkın oy dağılımı, favorinin payı, "
            "alan büyüklüğü, koşu tipi, hipodrom ve model–halk uyuşmazlığından hesaplanır.\n\n"
            "- 🟢 **Sakin** (<%55): favori güçlü, sürpriz bahisler geçmişte çok kaybettirdi.\n"
            "- 🟡 **Orta** (%55–70)\n- 🔴 **Kaotik** (>%70): sürpriz ikililerin geçmişte "
            "başabaşın üstünde kaldığı koşular.\n\n"
            "2025–26 testinde tahmin ↔ gerçek: Sakin %44↔%48, Orta %64↔%65, Kaotik %75↔%75.")


@st.cache_data(ttl=3600, show_spinner="Hipodromlar yükleniyor…")
def _sehirler(t: dt.date):
    return [(sid, ad) for sid, ad in scraper.get_cities(t) if scraper.is_domestic(ad)]


try:
    sehirler = _sehirler(tarih)
except Exception as e:
    st.error(f"TJK'ya ulaşılamadı: {e}")
    st.stop()
if not sehirler:
    st.warning("Bu tarihte yurt içi koşu bulunamadı.")
    st.stop()

secili = st.selectbox("Hipodrom", sehirler, format_func=lambda x: x[1])
sehir_id, sehir_adi = secili


@st.cache_resource(show_spinner="Geçmiş veri yükleniyor (ilk seferde uzun sürer)…")
def _gecmis():
    return gecmisi_yukle()


@st.cache_data(ttl=900, show_spinner="Program çekiliyor…")
def _program(t: dt.date, sehir: str, token: int):
    return gunun_programi(t, sehir)


@st.cache_resource(show_spinner="Özellikler ve model tahminleri hazırlanıyor…", max_entries=6)
def _tahminler(t_str: str, sehir: str, yeniden_egit: bool, token: int):
    t = dt.date.fromisoformat(t_str)
    prog = _program(t, sehir, token)
    if prog.empty:
        return None
    bugun, gecmis = ozellik_uret(_gecmis(), prog, t)
    if bugun.empty:
        return None
    feats = [c for c in FEATURE_COLS if c in gecmis.columns and c not in PIYASA_KOLONLARI]
    m_win, m_ilk2 = modelleri_hazirla(gecmis, feats, yeniden_egit)
    return tahmin_et(bugun, m_win, m_ilk2, feats), dt.datetime.now()


try:
    sonuc = _tahminler(tarih.isoformat(), sehir_adi, yeniden, ss.yenile)
except FileNotFoundError as e:
    st.error(f"{e}\n\nÖnce `python scripts/collect_history.py` çalıştırın.")
    st.stop()
if sonuc is None:
    st.warning("Bu hipodrom için program verisi alınamadı.")
    st.stop()
pred, uretim = sonuc
pred = pred.copy()
pred["tarih"] = tarih.isoformat()
kosular = sorted(pred["kosu_no"].dropna().unique())

# Her koşu için karışım olasılıkları ve kaos
K = {}
for k in kosular:
    try:
        d = v6.kosu_hazirla(pred[pred["kosu_no"] == k])
        if len(d) >= 2:
            K[int(k)] = {"d": d, "kaos": v6.kaos(d)}
    except Exception as e:  # tek koşunun verisi bozuksa diğerleri çalışsın
        st.caption(f"{int(k)}. koşu hesaplanamadı: {e}")
kosular = sorted(K)
st.caption(f"Son üretim: {uretim:%H:%M:%S} · AGF değiştikçe kaos seviyesi ve öneriler de değişir — "
           f"koşudan önce **yenile**.")


def _meta(d):
    r = d.iloc[0]
    parca = []
    for alan in ("saat", "mesafe", "pist", "kosu_turu"):
        v = r.get(alan)
        if pd.notna(v) and v != "":
            parca.append(f"{int(v)}m" if alan == "mesafe" else str(v))
    return " · ".join(parca)


def _nolu(d, nolar):
    m = d.set_index("no")["at"]
    return ", ".join(f"{int(n)} {m.get(n, '')}" for n in nolar if pd.notna(n))


sekme1, sekme2, sekme3 = st.tabs(["🎯 Koşular & İkili", "🧩 Altılı seçim aracı", "📊 Günün sonuçları"])

# =========================================================================== KOŞULAR
with sekme1:
    st.subheader("Günün kaos haritası")
    ozet = []
    for k in kosular:
        d, ka = K[k]["d"], K[k]["kaos"]
        o = v6.ikili_onerileri(d, agf_esigi, model_esigi)
        fav = d.loc[d.pa.idxmax()]
        ozet.append({"Koşu": k, "Bilgi": _meta(d), "At": len(d),
                     "Kaos": f"{ka['ikon']} {ka['seviye']}", "Favori kaybeder %": round(100 * ka["p"]),
                     "AGF favorisi": f"{int(fav['no'])} {fav['at']}" if pd.notna(fav["no"]) else fav["at"],
                     "Model 1.": f"{int(o['model1']['no'])} {o['model1']['at']}",
                     "Sürpriz adayı": _nolu(d, o["surpriz_atlar"]["no"]) or "—"})
    genis(st.dataframe, pd.DataFrame(ozet), hide_index=True,
                 column_config={"Favori kaybeder %": st.column_config.ProgressColumn(
                     "Favori kaybeder %", min_value=0, max_value=100, format="%d%%")})

    sadece = st.radio("Göster", ["Tüm koşular", "Sadece Orta + Kaotik", "Sadece Kaotik"], horizontal=True)
    for k in kosular:
        d, ka = K[k]["d"], K[k]["kaos"]
        if sadece == "Sadece Orta + Kaotik" and ka["seviye"] == "Sakin":
            continue
        if sadece == "Sadece Kaotik" and ka["seviye"] != "Kaotik":
            continue
        o = v6.ikili_onerileri(d, agf_esigi, model_esigi)
        baslik = f"{ka['ikon']} {k}. Koşu — {_meta(d)} — {ka['seviye']} (favori kaybeder %{100*ka['p']:.0f})"
        with st.expander(baslik, expanded=ka["seviye"] == "Kaotik"):
            notlar = surpriz_notlari(d, agf_esigi, model_esigi, 1.6)
            g = pd.DataFrame({
                "No": d["no"].astype("Int64"), "At": d["at"], "Jokey": d.get("Jokey"),
                "Ganyan": d.get("oran"), "AGF %": (100 * d.pa).round(1), "AGF sıra": d.agf_sira,
                "Model %": (100 * d.pm).round(1), "Model sıra": d.model_sira,
                "Karışım %": (100 * d.pk).round(1), "Not": notlar})
            genis(st.dataframe, g, hide_index=True,
                         column_config={"Karışım %": st.column_config.ProgressColumn(
                             "Karışım %", min_value=0, max_value=float(g["Karışım %"].max() or 1),
                             format="%.1f", help="Model + AGF karışımı: en kalibre kazanma olasılığı")})
            if d["agf_kaynak"].iloc[0] != "AGF":
                st.warning(f"AGF henüz yok — halk olasılığı {d['agf_kaynak'].iloc[0]} üzerinden hesaplandı.")

            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**🛡️ Olasılığı yüksek ikili** — normal kazanç")
                gt = o["guvenli"].assign(**{"Olasılık %": lambda x: (100 * x.p).round(1),
                                            "Tahmini ödeme": lambda x: x.odeme.round(1)})
                genis(st.dataframe, gt[["Çift", "Atlar", "Olasılık %", "Tahmini ödeme"]], hide_index=True)
                st.caption("Geçmiş test: AGF'nin ilk 2'si −%30, modelin ilk 2'si −%19 ROI. "
                           "Sık tutar, az öder.")
            with c2:
                st.markdown("**💥 Yüksek kazanç ikili** — Model 1. × sürpriz atlar")
                if o["yuksek"].empty:
                    st.info(f"Bu koşuda sürpriz adayı yok (halk sırası {agf_esigi}+ ve model ilk {model_esigi}).")
                else:
                    yt = o["yuksek"].assign(**{"Olasılık %": lambda x: (100 * x.p).round(1),
                                               "Tahmini ödeme": lambda x: x.odeme.round(1)})
                    genis(st.dataframe, yt[["Çift", "Atlar", "Olasılık %", "Tahmini ödeme"]], hide_index=True)
                    if ka["seviye"] == "Sakin":
                        st.warning("Sakin koşu: bu strateji geçmişte burada −%35 kaybettirdi.")
                    else:
                        st.caption("Geçmiş test (Orta+Kaotik, 2022–26): +%4,8 ROI, 249 isabet. "
                                   "İnce bir avantaj, garanti değil.")

            st.markdown("**🧮 Kendi ikili kolonun**")
            secenek = d["no"].dropna().astype(int).tolist()
            varsayilan = d.sort_values("pk", ascending=False)["no"].dropna().astype(int).head(5).tolist()
            sec = st.multiselect("Kolondaki atlar", secenek, default=varsayilan, key=f"kolon_{k}",
                                 format_func=lambda n, d=d: f"{n} — {d.set_index('no').at[n, 'at']}")
            kh = v6.kolon_hesapla(d, sec)
            if kh:
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Kombinasyon", kh["komb"])
                m2.metric("Kupon bedeli", f"{kh['komb'] * ikili_misli:,.0f} TL")
                m3.metric("Tutma olasılığı", f"%{100*kh['p']:.0f}")
                m4.metric("İsabette tipik kazanç", f"{kh['kat_med']:.1f}× bedel",
                          help="Kolondaki çiftlerin tahmini ödeme medyanı / kombinasyon sayısı")
                st.caption(f"Tahmini ödeme aralığı: {kh['odeme_min']:.0f} – {kh['odeme_max']:.0f} TL "
                           f"(1 TL için, ganyan oranlarından kaba tahmin). Geçmiş testte geniş kolonlar "
                           f"(6+ at) her kaos seviyesinde −%25…−%30 getirdi: geniş kolon, az ödeyen favori "
                           f"çiftlerini de satın alır.")

# =========================================================================== ALTILI
with sekme2:
    st.subheader("Altılı")
    st.caption("Hazır kuponlar model + AGF karışımıyla kurulur (belirsiz ayaklar genişler, net ayaklar banko "
               "geçilir). İsabet olasılığı da bu karışımla hesaplanır. Son kararı 'Kendi kuponun' bölümünde verirsin.")
    if len(kosular) < 6:
        st.warning("Bu hipodromda 6 ayak yok.")
    else:
        son = kosular[-1]
        secenekler = {f"{s}–{s+5}. koşular": s for s in kosular if s + 5 <= son}
        varsay = list(secenekler).index(f"{son-5}–{son}. koşular") if f"{son-5}–{son}. koşular" in secenekler else 0
        aralik = st.selectbox("Altılı ayakları", list(secenekler), index=varsay,
                              help="8 koşulu günlerde 1. altılı 1–6, 2. altılı 3–8. koşulardır.")
        ilk = secenekler[aralik]
        ayak_no = [k for k in kosular if k >= ilk][:6]
        ayaklar = [K[k]["d"] for k in ayak_no]
        anahtar = f"{tarih}|{sehir_adi}|{ilk}"

        def _yukle(oneri):
            for i, s in enumerate(oneri):
                ss[f"ayak_{anahtar}_{i}"] = [int(x) for x in s if pd.notna(x)]

        # --- Hazır kuponlar (v5 yapısı) -----------------------------------------------
        st.markdown("#### 📋 Hazır kuponlar")
        h1, h2, h3 = st.columns(3)
        hedef_dar = h1.number_input("Dar (kombinasyon)", 8, 2000, 48, 8, key=f"hd_{anahtar}")
        hedef_orta = h2.number_input("Orta (kombinasyon)", 50, 20000, 432, 50, key=f"ho_{anahtar}")
        hedef_genis = h3.number_input("Geniş (kombinasyon)", 200, 100000, 2592, 100, key=f"hg_{anahtar}")
        hazir = {ad: v6.altili_oneri(ayaklar, h) for ad, h in
                 (("Dar", hedef_dar), ("Orta", hedef_orta), ("Geniş", hedef_genis))}
        for kol, (ad, kup) in zip(st.columns(3), hazir.items()):
            oz = v6.altili_ozet(ayaklar, kup, misli)
            with kol:
                st.markdown(f"**{ad}**")
                a1, a2 = st.columns(2)
                a1.metric("Kombinasyon", f"{oz['komb']:,}")
                a2.metric("Bedel", f"{oz['maliyet']:,.0f} TL")
                a1.metric("İsabet olasılığı", f"%{100*oz['p']:.2f}")
                a2.metric("Halka göre", f"{oz['avantaj']:.2f}×")
                genis(st.dataframe, pd.DataFrame([
                    {"Ayak": f"{k}. Koşu {K[k]['kaos']['ikon']}", "At": len(s),
                     "Seçimler": " - ".join(str(int(n)) for n in s)}
                    for k, s in zip(ayak_no, kup)]), hide_index=True)
                if st.button(f"✏️ {ad} kuponunu düzenle", key=f"duz_{ad}_{anahtar}"):
                    _yukle(kup)

        alt_ad = st.radio("Tek at bırakılan ayaklarda alternatifler — kupon", list(hazir), index=1,
                          horizontal=True, key=f"altr_{anahtar}")
        alt = v6.tek_at_alternatifleri(ayaklar, hazir[alt_ad], ayak_no, misli)
        if alt.empty:
            st.caption(f"{alt_ad} kuponunda tek at bırakılan ayak yok.")
        else:
            genis(st.dataframe, alt, hide_index=True)
        with st.expander("Her ayakta yedek atlar (kümülatif kapsama)"):
            genis(st.dataframe, v6.yedek_at_tablosu(ayaklar, ayak_no), hide_index=True)

        # --- Kendi kuponun --------------------------------------------------------------
        st.divider()
        st.markdown("#### ✏️ Kendi kuponun")
        st.caption("Hazır kuponlardan birini 'düzenle' ile buraya al ya da sıfırdan seç.")
        b = st.columns(2)
        if b[0].button("Temizle", key=f"btn_tem_{anahtar}"):
            _yukle([[] for _ in ayaklar])
        if b[1].button("Banko (her ayağa 1 at)", key=f"btn_ban_{anahtar}"):
            _yukle(v6.altili_oneri(ayaklar, 1))

        secim = []
        kolonlar = st.columns(3)
        for i, (k, d) in enumerate(zip(ayak_no, ayaklar)):
            ka = K[k]["kaos"]
            with kolonlar[i % 3]:
                st.markdown(f"**{i+1}. ayak — {k}. koşu** {ka['ikon']} {ka['seviye']}")
                anahtar_i = f"ayak_{anahtar}_{i}"
                ss.setdefault(anahtar_i, [])
                lab = {int(r.no): f"{int(r.no)} {r.at} · %{100*r.pk:.0f} · AGF {r.agf_sira}."
                       for r in d.itertuples() if pd.notna(r.no)}
                s = st.multiselect("Atlar", list(lab), key=anahtar_i, format_func=lambda n, lab=lab: lab[n],
                                   label_visibility="collapsed")
                p_ayak = float(d[d["no"].isin(s)].pk.sum())
                st.progress(min(p_ayak, 1.0), text=f"{len(s)} at · ayağı kapsama %{100*p_ayak:.0f}")
                secim.append(s)

        st.divider()
        if all(secim):
            oz = v6.altili_ozet(ayaklar, secim, misli)
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Kombinasyon", f"{oz['komb']:,}")
            m2.metric("Kupon bedeli", f"{oz['maliyet']:,.2f} TL",
                      help=f"{oz['komb']:,} × {oz['birim']:.2f} TL × {misli} misli")
            m3.metric("İsabet olasılığı", f"%{100*oz['p']:.2f}")
            m4.metric("Halka göre", f"{oz['avantaj']:.2f}×",
                      help="Senin isabet olasılığın / halkın (AGF) isabet olasılığı. 1'in üstü: kupon halkın "
                           "küçümsediği atları içeriyor; tutarsa ikramiye ortalamanın üstünde olur.")
            kupon = " / ".join(",".join(str(n) for n in sorted(s)) for s in secim)
            st.code(kupon, language=None)
            zayif = [f"{ayak_no[i]}. koşu (%{100*float(a[a['no'].isin(s)].pk.sum()):.0f})"
                     for i, (a, s) in enumerate(zip(ayaklar, secim))
                     if float(a[a["no"].isin(s)].pk.sum()) < 0.5]
            if zayif:
                st.warning("Kuponu en çok riske atan ayaklar: " + ", ".join(zayif))
            alt2 = v6.tek_at_alternatifleri(ayaklar, secim, ayak_no, misli)
            if not alt2.empty:
                st.markdown("**Tek at yazdığın ayaklarda alternatifler**")
                genis(st.dataframe, alt2, hide_index=True)
        else:
            st.info("Her ayaktan en az bir at seç (ya da bir öneri butonuna bas).")
        st.caption("Geçmiş test (5.862 altılı, gerçek ikramiyeler): altılı havuzu bahisçiye yaklaşık %48 "
                   "geri veriyor; hiçbir kupon stratejisi uzun vadede kâr etmedi. Eğlence bütçesiyle oyna.")

# =========================================================================== SONUÇLAR
with sekme3:
    @st.cache_data(ttl=600, show_spinner="Sonuçlar çekiliyor…")
    def _sonuclar(t: dt.date, liste: tuple):
        parca = [sonuclar.gunluk_sonuclar(t, ad, sid) for sid, ad in liste]
        parca = [p for p in parca if not p.empty]
        return pd.concat(parca, ignore_index=True) if parca else pd.DataFrame()

    e1, e2 = st.columns([1, 2])
    tum = e2.checkbox("Tüm yurt içi hipodromlar", value=False)
    eda_anahtar = f"eda|{tarih}|{sehir_adi}|{tum}"
    if e1.button("📊 Sonuçları getir"):
        ss[eda_anahtar] = True
    if ss.get(eda_anahtar):
        sdf = _sonuclar(tarih, tuple(sehirler) if tum else (secili,))
        if sdf.empty:
            st.info("Sonuç bulunamadı — koşular bitmemiş ya da sonuç dosyası yayınlanmamış olabilir.")
        else:
            sdf = eda.hazirla(sdf)
            w = eda.kazananlar(sdf, agf_esigi=agf_esigi)
            oz = eda.ozet(sdf, w)
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Koşu", oz["kosu"])
            k2.metric("Favori kazandı", "-" if pd.isna(oz["fav_kazanma"]) else f"%{100*oz['fav_kazanma']:.0f}")
            k3.metric("Kazanan ganyan (medyan)", "-" if pd.isna(oz["ort_ganyan"]) else f"{oz['ort_ganyan']:.2f}")
            k4.metric("Sürpriz kazanan", oz["surpriz"])
            if not tum:
                kaos_kosu = pd.DataFrame([{"kosu_no": k, "Kaos": f"{K[k]['kaos']['ikon']} {K[k]['kaos']['seviye']}"}
                                          for k in kosular])
                w = w.assign(kosu_no=pd.to_numeric(w["kosu_no"], errors="coerce")).merge(kaos_kosu, on="kosu_no", how="left")
            genis(st.dataframe, w.rename(columns={
                "hipodrom": "Hipodrom", "kosu_no": "Koşu", "saat": "Saat", "mesafe": "Mesafe", "pist": "Pist",
                "no": "No", "at": "At", "jokey": "Jokey", "antrenor": "Antrenör", "sahip": "Sahip",
                "ganyan": "Ganyan", "agf": "AGF", "agf_sira": "AGF sıra", "derece": "Derece",
                "surpriz": "Sürpriz"}), hide_index=True)
            g1 = eda.kazanan_ganyan_grafik(w)
            if g1 is not None:
                genis(st.altair_chart, g1)
            for sek, (kol, ad) in zip(st.tabs(["Jokey", "Antrenör", "Sahip"]),
                                      [("jokey", "Jokey"), ("antrenor", "Antrenör"), ("sahip", "Sahip")]):
                with sek:
                    t = eda.kisi_tablosu(sdf, kol)
                    if t.empty:
                        st.caption(f"{ad} bilgisi sonuç dosyasında yok.")
                        continue
                    a, b2 = st.columns([3, 2])
                    genis(a.altair_chart, eda.kisi_grafik(t, kol, f"{ad} — günün ilk 3'leri"))
                    genis(b2.dataframe, t.rename(columns={kol: ad, "binis": "Biniş", "birinci": "1.", "ilk3": "İlk 3",
                                                   "kazanma_%": "Kazanma %"}).drop(columns="ikinci_ucuncu"),
                                 hide_index=True, height=420)
            st.download_button("CSV indir", sdf.to_csv(index=False).encode("utf-8-sig"),
                               file_name=f"sonuclar_{tarih:%Y%m%d}.csv", mime="text/csv")
    else:
        st.info("Seçili günün sonuçlarını görmek için **Sonuçları getir**'e bas.")

st.divider()
st.caption("Kaynak testler: scripts/surpriz_backtest.py, ikili_kolon_backtest.py, altili_gercek.py "
           "(2022–2026 gerçek TJK ödemeleri). Nihai karar senin. Sorumlu oyna.")
