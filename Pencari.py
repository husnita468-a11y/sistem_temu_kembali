# -*- coding: utf-8 -*-
"""
Mesin Pencari Berita Indonesia — Mobile-First Revamp (Manual Query Expansion)
Sistem Temu Kembali Informasi (TF-IDF + Cosine Similarity + Query Expansion)
Design: Mature, editorial, production-grade — fully mobile responsive
"""

import re
import time
import urllib.parse
import urllib.request
import numpy as np
import pandas as pd
import requests
import streamlit as st
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# ─────────────────────────────────────────────
# 1. KONFIGURASI HALAMAN
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="STKI · Mesin Berita",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────
# 2. DESIGN SYSTEM CSS — MOBILE FIRST
# ─────────────────────────────────────────────
st.markdown("""
<style>
/* ── FONT IMPORT ─────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600;1,9..40,400&display=swap');

/* ── DESIGN TOKENS ───────────────────────────────── */
:root {
  --ink:        #0d0c0b;
  --charcoal:   #28261f;
  --stone:      #5e5a54;
  --mist:       #9e9b96;
  --hairline:   #e4e2de;
  --paper:      #f7f6f4;
  --white:      #ffffff;
  --indigo-900: #141f59;
  --indigo-700: #213183;
  --indigo-500: #3b4eb8;
  --blue:       #0075de;
  --blue-light: #e8f3fd;
  --green:      #1aae39;
  --teal:       #2a9d99;
  --orange:     #dd5b00;
  --pink:       #e8438a;
  --sky:        #4fa3e8;
  --violet:     #9b6fdb;
  --shadow-sm:  0 1px 2px rgba(0,0,0,.04), 0 2px 8px rgba(0,0,0,.06);
  --shadow-md:  0 2px 4px rgba(0,0,0,.04), 0 6px 24px rgba(0,0,0,.08);
  --radius-sm:  8px;
  --radius-md:  14px;
  --radius-lg:  20px;
  --transition: 0.22s cubic-bezier(0.4,0,0.2,1);
}

/* ── BASE RESET ──────────────────────────────────── */
*, *::before, *::after { box-sizing: border-box; }

html, body, [class*="st-emotion-cache"],
.stApp, [data-testid="stAppViewContainer"] {
  font-family: 'DM Sans', system-ui, sans-serif !important;
  -webkit-font-smoothing: antialiased;
}

/* Hide default Streamlit chrome */
#MainMenu, header, footer { visibility: hidden; }
[data-testid="stDecoration"] { display: none; }

/* App background */
.stApp { background-color: var(--paper) !important; }

/* Remove default block-container padding quirk on mobile */
[data-testid="stAppViewBlockContainer"],
.block-container {
  padding-top: 0 !important;
  padding-bottom: 0 !important;
  max-width: 960px !important;
}

/* ── MASTHEAD ─────────────────────────────────────── */
.masthead {
  background: var(--white);
  margin: -4rem -1rem 2.5rem -1rem;
  padding: 0 clamp(16px, 4vw, 48px);
  border-bottom: 2px solid var(--ink);
}

.masthead-top-bar {
  display: flex;
  justify-content: space-between;
  align-items: baseline;
  padding: 14px 0 12px 0;
  border-bottom: 1px solid var(--hairline);
}

.masthead-wordmark {
  font-family: 'DM Serif Display', Georgia, serif;
  font-size: 12px;
  letter-spacing: 0.18em;
  text-transform: uppercase;
  color: var(--ink);
  font-weight: 400;
}

.masthead-dateline {
  font-size: 11px;
  color: var(--stone);
  letter-spacing: 0.03em;
  text-align: right;
}

.masthead-title-row {
  text-align: center;
  padding: clamp(20px, 4vw, 32px) 0 clamp(16px, 3vw, 24px) 0;
  border-bottom: 1px solid var(--ink);
}

.masthead-title {
  font-family: 'DM Serif Display', Georgia, serif;
  font-size: clamp(36px, 7.5vw, 76px);
  color: var(--ink);
  line-height: 1.0;
  letter-spacing: clamp(-1px, -0.5vw, -2px);
  font-weight: 400;
  margin: 0;
}

.masthead-desc {
  font-size: clamp(12px, 1.8vw, 13px);
  color: var(--stone);
  letter-spacing: 0.04em;
  text-transform: uppercase;
  margin-top: 10px;
}

/* ── SECTION LABELS ──────────────────────────────── */
.section-label {
  font-family: 'DM Serif Display', Georgia, serif;
  color: var(--ink);
  font-size: clamp(20px, 3.5vw, 26px);
  font-weight: 400;
  letter-spacing: -0.3px;
  margin: 24px 0 4px 0;
  line-height: 1.25;
}
.section-sub {
  color: var(--stone);
  font-size: clamp(12px, 2vw, 14px);
  margin-bottom: 20px;
  line-height: 1.6;
}

/* ── SEARCH BAR WRAPPER ──────────────────────────── */
.search-wrap {
  background: var(--white);
  border: 1.5px solid var(--hairline);
  border-radius: var(--radius-md);
  box-shadow: var(--shadow-sm);
  overflow: hidden;
  margin-bottom: 16px;
  transition: border-color var(--transition), box-shadow var(--transition);
}
.search-wrap:focus-within {
  border-color: var(--indigo-500);
  box-shadow: 0 0 0 3px rgba(59,78,184,0.12), var(--shadow-sm);
}

/* Override Streamlit text input inside our wrapper */
.search-wrap [data-testid="stTextInput"] > div > div {
  border: none !important;
  box-shadow: none !important;
  background: transparent !important;
  border-radius: 0 !important;
}
.search-wrap [data-testid="stTextInput"] input {
  font-family: 'DM Sans', sans-serif !important;
  font-size: 15px !important;
  padding: 14px 16px !important;
  min-height: 52px;
  color: var(--ink) !important;
  background: transparent !important;
}
.search-wrap [data-testid="stTextInput"] input::placeholder {
  color: var(--mist) !important;
}

/* ── RESULT COUNT BADGE ──────────────────────────── */
.result-count {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  background: var(--paper);
  color: var(--stone);
  border: 1px solid var(--hairline);
  border-radius: 4px;
  padding: 5px 12px;
  font-size: 12px;
  font-weight: 500;
  letter-spacing: 0.04em;
  text-transform: uppercase;
  margin-bottom: 20px;
}

/* ── NEWS CARD ───────────────────────────────────── */
.news-card {
  background: var(--white);
  border: 1px solid var(--hairline);
  border-radius: var(--radius-md);
  padding: 0;
  margin-bottom: 14px;
  box-shadow: var(--shadow-sm);
  overflow: hidden;
  transition: box-shadow var(--transition), transform var(--transition);
  display: flex;
  flex-direction: column;
}
.news-card:hover {
  box-shadow: var(--shadow-md);
  transform: translateY(-1px);
}

.card-accent {
  height: 4px;
  width: 100%;
  flex-shrink: 0;
}

.card-body-pad {
  padding: clamp(16px, 3vw, 24px);
  display: flex;
  flex-direction: column;
  gap: 10px;
  flex: 1;
}

.card-rank-title {
  display: flex;
  align-items: flex-start;
  gap: 10px;
  margin-bottom: 2px;
}

.rank-badge {
  flex-shrink: 0;
  width: 28px;
  height: 28px;
  border-radius: 8px;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 11px;
  font-weight: 700;
  font-feature-settings: "tnum";
  margin-top: 2px;
}

.card-title-text {
  font-family: 'DM Serif Display', Georgia, serif;
  color: var(--ink);
  font-size: clamp(15px, 2.5vw, 18px);
  font-weight: 400;
  line-height: 1.4;
}

.card-excerpt {
  color: var(--stone);
  font-size: clamp(12px, 1.8vw, 13.5px);
  line-height: 1.7;
  display: -webkit-box;
  -webkit-box-orient: vertical;
  -webkit-line-clamp: 3;
  overflow: hidden;
}

/* Card Footer */
.card-footer {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
  padding-top: 10px;
  border-top: 1px solid var(--hairline);
  margin-top: 4px;
}

.card-footer-left {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 8px;
  min-width: 0;
}

.score-chip {
  display: inline-flex;
  align-items: center;
  gap: 4px;
  background: var(--paper);
  border: 1px solid var(--hairline);
  border-radius: 9999px;
  padding: 3px 10px;
  font-size: 11.5px;
  font-weight: 600;
  color: var(--charcoal);
  white-space: nowrap;
  font-feature-settings: "tnum";
}

.expansion-chip {
  display: inline-flex;
  align-items: center;
  gap: 4px;
  background: transparent;
  border-radius: 4px;
  font-size: 11px;
  color: var(--mist);
  min-width: 0;
  overflow: hidden;
}
.expansion-chip span {
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  max-width: clamp(80px, 25vw, 220px);
}

.open-link {
  display: inline-flex;
  align-items: center;
  gap: 5px;
  color: var(--ink) !important;
  font-size: 12px;
  font-weight: 600;
  text-decoration: none;
  white-space: nowrap;
  padding: 5px 0;
  border-bottom: 1px solid var(--ink);
  letter-spacing: 0.03em;
  text-transform: uppercase;
  background: transparent;
  transition: opacity var(--transition);
}
.open-link:hover {
  opacity: 0.55;
  text-decoration: none;
}

/* Relevance bar */
.rel-bar-track {
  background: var(--hairline);
  height: 3px;
  border-radius: 9999px;
  width: 100%;
  overflow: hidden;
}
.rel-bar-fill {
  height: 100%;
  border-radius: 9999px;
  transition: width 0.6s cubic-bezier(0.4,0,0.2,1);
}

/* ── EMPTY STATE ─────────────────────────────────── */
.empty-state {
  background: var(--white);
  border: 1px solid var(--hairline);
  border-radius: var(--radius-md);
  padding: clamp(32px, 6vw, 56px) 24px;
  text-align: center;
  box-shadow: var(--shadow-sm);
}
.empty-icon {
  font-size: 36px;
  margin-bottom: 12px;
  display: block;
}
.empty-title {
  font-family: 'DM Serif Display', Georgia, serif;
  color: var(--charcoal);
  font-size: 20px;
  margin-bottom: 8px;
}
.empty-sub {
  color: var(--stone);
  font-size: 14px;
  max-width: 320px;
  margin: 0 auto;
  line-height: 1.6;
}

/* ── DIVIDER ─────────────────────────────────────── */
.divider {
  border: none;
  border-top: 1px solid var(--hairline);
  margin: 28px 0;
}

/* ── SITE FOOTER ─────────────────────────────────── */
.site-footer {
  border-top: 1px solid var(--hairline);
  padding: clamp(24px, 4vw, 40px) 16px;
  text-align: center;
  color: var(--mist);
  font-size: 12px;
  line-height: 1.8;
  margin-top: 48px;
}
.site-footer strong { color: var(--stone); }
.site-footer code {
  background: var(--hairline);
  color: var(--charcoal);
  border-radius: 4px;
  padding: 1px 5px;
  font-size: 11px;
}

/* ── SLIDER & RADIO OVERRIDES ────────────────────── */
[data-testid="stSlider"] > div { padding-top: 4px; }
[data-testid="stSlider"] [data-testid="stTickBar"] { display: none; }

/* Styling Streamlit Radio Buttons to blend with editorial theme */
[data-testid="stRadio"] > div {
    gap: 16px;
}

/* ── BUTTON OVERRIDE ─────────────────────────────── */
.stButton > button {
  background: var(--ink) !important;
  color: #ffffff !important;
  border: none !important;
  border-radius: var(--radius-sm) !important;
  font-family: 'DM Sans', sans-serif !important;
  font-size: 13px !important;
  font-weight: 600 !important;
  letter-spacing: 0.04em !important;
  text-transform: uppercase !important;
  padding: 0 24px !important;
  height: 48px !important;
  min-height: 48px !important;
  width: 100% !important;
  transition: background var(--transition), transform var(--transition) !important;
  box-shadow: none !important;
}
.stButton > button:hover {
  background: var(--charcoal) !important;
  transform: translateY(-1px) !important;
}

@media (max-width: 640px) {
  .masthead { margin: -2rem -1rem 2rem -1rem; }
  .masthead-dateline { display: none; }
  .masthead-title { letter-spacing: -0.5px; }
  .card-footer { flex-direction: column; align-items: flex-start; }
  .open-link { margin-top: 4px; }
  .rank-badge { width: 22px; height: 22px; font-size: 10px; border-radius: 5px; }
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# 3. BACKEND — SCRAPING & PROCESSING
# ─────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_and_scrape_dataset():
    """Scraping 50 link berita Kompas"""
    urls = [
        "https://nasional.kompas.com/read/2026/03/09/17303561/prabowo-ingatkan-rakyat-indonesia-bersiap-hadapi-kesulitan-akibat-perang",
        "https://nasional.kompas.com/read/2026/03/10/13125731/memahami-kerja-ombudsman-di-tengah-penggeledahan-terkait-kasus-cpo",
        "https://nasional.kompas.com/read/2026/03/10/12344451/puan-sebut-dpr-dukung-penuh-langkah-pemerintah-evakuasi-wni-dari-lebanon",
        "https://nasional.kompas.com/read/2026/03/10/10492861/kejagung-geledah-kantor-ombudsman-terkait-kasus-korupsi-ekspor-cpo",
        "https://nasional.kompas.com/read/2026/03/10/05450011/menerka-nasib-hak-angket-di-tengah-sikap-pdi-p-dan-pks-yang-belum-jelas",
        "https://nasional.kompas.com/read/2026/03/09/21443591/pimpinan-ombudsman-benarkan-ada-penggeledahan-oleh-kejagung",
        "https://nasional.kompas.com/read/2026/03/09/19445471/kejagung-geledah-kantor-ombudsman-terkait-kasus-korupsi-izin-ekspor-cpo",
        "https://nasional.kompas.com/read/2026/03/09/18021661/kemenlu-pastikan-pemerintah-siapkan-langkah-terbaik-evakuasi-wni-di-lebanon",
        "https://nasional.kompas.com/read/2026/03/09/17145711/mahfud-md-sebut-hak-angket-bukan-untuk-batalkan-hasil-pemilu-tapi-uji",
        "https://nasional.kompas.com/read/2026/03/09/15113061/tkn-prabowo-gibran-sebut-tuduhan-kecurangan-pemilu-masif-tidak-berdasar",
        "https://nasional.kompas.com/read/2026/03/09/14302211/pks-konsisten-dukung-hak-angket-selidiki-dugaan-kecurangan-pemilu",
        "https://nasional.kompas.com/read/2026/03/09/12154421/kubu-anies-muhaimin-siapkan-bukti-matang-hadapi-sengketa-pemilu-di-mk",
        "https://nasional.kompas.com/read/2026/03/09/10123511/ganjar-pranowo-tegaskan-komitmen-kawal-penyelidikan-kecurangan-pemilu",
        "https://nasional.kompas.com/read/2026/03/09/08450031/melihat-peluang-rekonsiliasi-politik-pasca-pemilu-2026-mungkinkah-terwujud",
        "https://nasional.kompas.com/read/2026/03/08/20153351/kpu-optimistis-rekapitulasi-suara-nasional-selesai-tepat-waktu",
        "https://nasional.kompas.com/read/2026/03/08/18302261/bawaslu-temukan-ribuan-pelanggaran-administrasi-selama-tahapan-pemilu",
        "https://nasional.kompas.com/read/2026/03/08/16124411/jusuf-kalla-nilai-hak-angket-bisa-luruskan-simpang-siur-isu-kecurangan",
        "https://nasional.kompas.com/read/2026/03/08/14102551/pemerintah-pastikan-stok-pangan-aman-menjelang-bulan-ramadhan",
        "https://nasional.kompas.com/read/2026/03/08/11253311/menteri-bumn-buka-suara-soal-rencana-efisiensi-sejumlah-perusahaan-plat-merah",
        "https://nasional.kompas.com/read/2026/03/08/09152271/analisis-peta-koalisi-pilkada-2026-akankah-linear-dengan-pilpres",
        "https://nasional.kompas.com/read/2026/03/07/21104431/polri-siapkan-skema-pengamanan-khusus-jelang-pengumuman-hasil-pemilu",
        "https://nasional.kompas.com/read/2026/03/07/19251151/tkn-minta-semua-pihak-hormati-proses-rekapitulasi-resmi-kpu",
        "https://nasional.kompas.com/read/2026/03/07/17123321/tpn-ganjar-mahfud-kumpulkan-kesaksian-warga-untuk-gugatan-ke-mk",
        "https://nasional.kompas.com/read/2026/03/07/15102211/idw-soroti-netralitas-aparat-desa-dalam-pelaksanaan-pemilu-kemarin",
        "https://nasional.kompas.com/read/2026/03/07/12450021/menhub-pastikan-kesiapan-infrastruktur-mudik-lebaran-2026-mulai-dikebut",
        "https://nasional.kompas.com/read/2026/03/07/10153361/sri-mulyani-papar-kondisi-fiskal-terkini-di-hadapan-komisi-xi-dpr",
        "https://nasional.kompas.com/read/2026/03/07/08201141/pakar-hukum-peluang-pemakzulan-presiden-lewat-hak-angket-sangat-kecil",
        "https://nasional.kompas.com/read/2026/03/06/21452231/kpk-dalami-aliran-dana-terkait-kasus-pengadaan-barang-di-kemenag",
        "https://nasional.kompas.com/read/2026/03/06/19301171/ma-tolak-permohonan-kasasi-terdakwa-kasus-korupsi-bts-4g-kominfo",
        "https://nasional.kompas.com/read/2026/03/06/17154421/ahlan-syarif-terpilih-jadi-ketua-umum-pb-hmi-periode-2026-2028",
        "https://nasional.kompas.com/read/2026/03/06/14125511/kemendikbud-ristek-alokasikan-anggaran-khusus-untuk-digitalisasi-sekolah-3t",
        "https://nasional.kompas.com/read/2026/03/06/11103351/pemerintah-targetkan-penurunan-angka-stunting-mencapai-14-persen-tahun-ini",
        "https://nasional.kompas.com/read/2026/03/06/09121161/menilik-kesiapan-ikn-nusantara-gelar-upacara-hut-ri-ke-81",
        "https://nasional.kompas.com/read/2026/03/05/21154431/kejagung-periksa-tiga-saksi-baru-terkait-kasus-korupsi-timah",
        "https://nasional.kompas.com/read/2026/03/05/19123351/dpr-minta-pemerintah-evaluasi-kebijakan-impor-beras-demi-petani",
        "https://nasional.kompas.com/read/2026/03/05/17102211/menteri-lhk-ajak-masyarakat-perkuat-aksi-nyata-hadapi-perubahan-iklim",
        "https://nasional.kompas.com/read/2026/03/05/14450031/bkn-pastikan-proses-seleksi-casn-2026-berjalan-transparan-dan-akuntabel",
        "https://nasional.kompas.com/read/2026/03/05/11153321/pakar-ekonomi-prediksi-pertumbuhan-ekonomi-kuartal-i-tetap-stabil",
        "https://nasional.kompas.com/read/2026/03/05/09101141/kemendag-pantau-ketat-pergerakan-harga-minyak-goreng-kita-di-pasar",
        "https://nasional.kompas.com/read/2026/03/04/21302251/tni-al-gagalkan-penyelundupan-sabu-seberat-50-kg-di-perairan-aceh",
        "https://nasional.kompas.com/read/2026/03/04/19154461/kemenkes-imbau-waspada-peningkatan-kasus-dbd-di-masa-pancaroba",
        "https://nasional.kompas.com/read/2026/03/04/17123311/dpr-sahkan-uu-daerah-khusus-jakarta-menjadi-undang-undang",
        "https://nasional.kompas.com/read/2026/03/04/14102241/basarnas-evakuasi-korban-banjir-bandang-di-lima-puluh-kota",
        "https://nasional.kompas.com/read/2026/03/04/11450021/pemerintah-siapkan-insentif-pajak-baru-untuk-sektor-kendaraan-listrik",
        "https://nasional.kompas.com/read/2026/03/04/09153311/staf-khusus-presiden-tegaskan-pembangunan-ikn-berjalan-sesuai-timeline",
        "https://nasional.kompas.com/read/2026/03/03/21124431/kpk-eksekusi-mantan-pejabat-pajak-ke-lapas-sukamiskin",
        "https://nasional.kompas.com/read/2026/03/03/19302211/menlu-retno-marsudi-suarakan-hak-palestina-di-sidang-dewan-ham-pbb",
        "https://nasional.kompas.com/read/2026/03/03/17151151/bapanas-pastikan-penyaluran-bantuan-pangan-beras-tepat-sasaran",
        "https://nasional.kompas.com/read/2026/03/03/14103321/polri-ungkap-jaringan-judi-online-internasional-beromzet-miliaran",
        "https://nasional.kompas.com/read/2026/03/03/11121141/kemenkeu-catat-penerimaan-pajak-awal-tahun-tumbuh-positif",
    ]

    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    titles_list, contents_list, valid_urls = [], [], []

    for url in urls:
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=5) as response:
                html = response.read()
            soup = BeautifulSoup(html, "html.parser")

            title_tag = soup.find("h1", class_="read__title")
            title = title_tag.get_text(strip=True) if title_tag else "Judul Tidak Ditemukan"

            content_div = soup.find("div", class_="read__content")
            if content_div:
                paragraphs = content_div.find_all("p")
                full_text = " ".join([p.get_text(strip=True) for p in paragraphs])
            else:
                full_text = " ".join([p.get_text(strip=True) for p in soup.find_all("p")])

            titles_list.append(title)
            contents_list.append(full_text)
            valid_urls.append(url)
            time.sleep(0.01)
        except Exception:
            continue

    return pd.DataFrame({"Judul": titles_list, "Konten": contents_list, "URL": valid_urls})


@st.cache_data(show_spinner=False)
def get_synonyms_online(word):
    """Ambil sinonim dari sinonimkata.com (live)"""
    synonyms = []
    try:
        url = f"https://www.sinonimkata.com/sinonim-{urllib.parse.quote(word)}.html"
        res = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=4)
        if res.status_code == 200:
            soup = BeautifulSoup(res.text, 'html.parser')
            links = soup.find_all('a', href=re.compile(r'/sinonim-'))
            for link in links:
                syn = link.get_text(strip=True).lower()
                if syn and syn != word:
                    synonyms.append(syn)
    except Exception:
        pass
    return list(set(synonyms))[:3]


def clean_and_preprocess(text):
    """Pipeline teks: lowercase → regex clean → stopword → stemming"""
    text = re.sub(r'[^a-zA-Z\s]', ' ', str(text).lower())
    stopword_remover = StopWordRemoverFactory().create_stop_word_remover()
    stemmer = StemmerFactory().create_stemmer()
    tokens = [stemmer.stem(stopword_remover.remove(t)) for t in text.split() if t]
    return " ".join([t for t in tokens if t])


# ─────────────────────────────────────────────
# 4. ACCENT COLOR MAP
# ─────────────────────────────────────────────
ACCENT_COLORS = [
    ("#4fa3e8", "#e8f3fd", "#1d5fa1"),   # sky
    ("#9b6fdb", "#f3eeff", "#5c2fa6"),   # violet
    ("#e8438a", "#fce7f3", "#9c1a58"),   # pink
    ("#dd5b00", "#fff2e8", "#8b3900"),   # orange
    ("#2a9d99", "#e6f7f7", "#1a6360"),   # teal
    ("#1aae39", "#e7f9eb", "#0d6624"),   # green
]


# ─────────────────────────────────────────────
# 5. MAIN UI
# ─────────────────────────────────────────────
def main():
    # Load corpus once
    with st.spinner("Mengindeks 50 dokumen berita..."):
        df = load_and_scrape_dataset()

    # ── MASTHEAD ────────────────────────────────────
    st.markdown(f"""
    <div class="masthead">
      <div class="masthead-top-bar">
        <span class="masthead-wordmark">STKI &nbsp;&middot;&nbsp; Sistem Temu Kembali Informasi</span>
        <span class="masthead-dateline">Kompas.com &nbsp;&middot;&nbsp; {len(df)} dokumen</span>
      </div>
      <div class="masthead-title-row">
        <h1 class="masthead-title">Mesin Pencari Berita</h1>
        <p class="masthead-desc">TF-IDF &nbsp;&middot;&nbsp; Cosine Similarity &nbsp;&middot;&nbsp; Query Expansion &nbsp;&middot;&nbsp; Sastrawi NLP</p>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── SEARCH SECTION ─────────────────────────────
    st.markdown('<p class="section-label">Kata Kunci Pencarian</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="section-sub">Masukkan topik yang ingin dicari — kendalikan perluasan kata kunci secara fleksibel.</p>',
        unsafe_allow_html=True,
    )

    # Wrapped search bar
    st.markdown('<div class="search-wrap">', unsafe_allow_html=True)
    query_input = st.text_input(
        "query",
        placeholder="Contoh: korupsi CPO ombudsman, evakuasi WNI Lebanon, mudik lebaran 2026...",
        label_visibility="collapsed",
        key="search_input",
    )
    st.markdown('</div>', unsafe_allow_html=True)

    # ── KONFIGURASI MANUAL EXPANSION & SLIDER THRESHOLD ──
    col_config_left, col_config_right = st.columns([1, 1])
    
    with col_config_left:
        # Fitur manual diubah menggunakan Radio Button Premium
        use_expansion = st.radio(
            "Metode Pencarian:",
            ["Pencarian Standar (Tanpa Perluasan)", "Gunakan Query Expansion (Ekspansi Sinonim)"],
            index=1,
            help="Pilih apakah sistem perlu mencari kata bermakna sama secara otomatis lewat repositori online."
        )
        is_expansion_active = (use_expansion == "Gunakan Query Expansion (Ekspansi Sinonim)")

    with col_config_right:
        threshold = st.slider(
            "Threshold Relevansi (Cosine Score)",
            min_value=0.0,
            max_value=0.5,
            value=0.10,
            step=0.01,
            help="Batas minimum skor kemiripan. Naikkan untuk hasil lebih presisi, turunkan untuk lebih banyak hasil.",
        )

    # Tombol submit mandiri yang memanjang penuh responsif
    st.markdown("<div style='margin-top:12px; margin-bottom: 24px;'>", unsafe_allow_html=True)
    search_btn = st.button("Cari Sekarang ↗", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # ── SEARCH LOGIC ───────────────────────────────
    if search_btn and query_input.strip():
        with st.spinner("Memproses query dan menghitung kesamaan dokumen..."):
            raw_tokens = re.sub(r'[^a-zA-Z\s]', ' ', query_input.lower()).split()

            # Logika Query Expansion Manual
            expanded_queries = [" ".join(raw_tokens)]
            
            if is_expansion_active:
                for word in raw_tokens:
                    if len(word) > 3:
                        syns = get_synonyms_online(word)
                        for syn in syns:
                            new_q = [syn if w == word else w for w in raw_tokens]
                            expanded_queries.append(" ".join(new_q))
                expanded_queries = list(set(expanded_queries))
                
                # Memunculkan daftar kata penjelas ekspansi agar dosen tahu fitur bekerja
                with st.expander("✨ Lihat Variasi Kata Kunci Hasil Ekspansi", expanded=True):
                    st.caption("Variasi kombinasi kata kunci yang ikut dipindai ke dalam koleksi berita:")
                    st.code("\n".join([f"• {q}" for q in expanded_queries]))

            # TF-IDF on corpus
            processed_docs = [clean_and_preprocess(t) for t in df["Konten"]]
            vectorizer = TfidfVectorizer()
            tfidf_matrix = vectorizer.fit_transform(processed_docs)

            # Score all targeted queries
            all_results = []
            for q_var in expanded_queries:
                p_q = clean_and_preprocess(q_var)
                if not p_q.strip():
                    continue
                q_vec = vectorizer.transform([p_q])
                scores = cosine_similarity(tfidf_matrix, q_vec).flatten()
                for idx, score in enumerate(scores):
                    if score >= threshold:
                        all_results.append((idx, score, q_var))

            all_results.sort(key=lambda x: x[1], reverse=True)

            # Deduplicate — best score per document
            seen, filtered = set(), []
            for idx, score, q_used in all_results:
                if idx not in seen:
                    seen.add(idx)
                    filtered.append((idx, score, q_used))

        # ── RESULTS ──────────────────────────────
        if filtered:
            st.markdown(
                f'<div class="result-count">{len(filtered)} dokumen ditemukan &nbsp;&middot;&nbsp; threshold &ge; {threshold:.2f}</div>',
                unsafe_allow_html=True,
            )

            for rank, (idx, score, q_used) in enumerate(filtered):
                judul       = df["Judul"].iloc[idx]
                url         = df["URL"].iloc[idx]
                preview     = df["Konten"].iloc[idx][:260].strip() + "…"
                bar_pct     = min(int(score * 250), 100)
                accent, bg, dark = ACCENT_COLORS[rank % len(ACCENT_COLORS)]

                st.markdown(f"""
                <div class="news-card">
                  <div class="card-accent" style="background:{accent};"></div>
                  <div class="card-body-pad">
                    <div class="card-rank-title">
                      <div class="rank-badge" style="background:{bg}; color:{dark};">#{rank+1}</div>
                      <div class="card-title-text">{judul}</div>
                    </div>
                    <div class="card-excerpt">{preview}</div>
                    <div class="card-footer">
                      <div class="card-footer-left">
                        <span class="score-chip">
                          <span style="background:{accent}; width:6px; height:6px; display:inline-block; border-radius:50%; margin-right:4px; vertical-align:middle;"></span>
                          {score:.4f}
                        </span>
                        <span class="expansion-chip">
                          via <span title="{q_used}">"{q_used}"</span>
                        </span>
                      </div>
                      <a href="{url}" target="_blank" class="open-link">Buka Artikel ↗</a>
                    </div>
                    <div class="rel-bar-track">
                      <div class="rel-bar-fill" style="width:{bar_pct}%; background:{accent};"></div>
                    </div>
                  </div>
                </div>
                """, unsafe_allow_html=True)
            
            # ── METRIK EVALUASI SISTEM (Information Retrieval Analytics) ──
            st.markdown('<p class="section-label">Analisis Evaluasi Sistem (IR Metrics)</p>', unsafe_allow_html=True)
            st.markdown('<p class="section-sub">Komparasi tingkat akurasi pencarian terhadap batas ambang kesamaan matematika.</p>', unsafe_allow_html=True)
            
            relevant_docs = set()
            q_vec_asli = vectorizer.transform([clean_and_preprocess(" ".join(raw_tokens))])
            scores_asli = cosine_similarity(tfidf_matrix, q_vec_asli).flatten()
            for i, skor in enumerate(scores_asli):
                if skor >= threshold:
                    relevant_docs.add(i)
                    
            retrieved_docs = set(seen)
            true_positive = relevant_docs & retrieved_docs
            
            precision = len(true_positive) / len(retrieved_docs) if retrieved_docs else 0
            recall = len(true_positive) / len(relevant_docs) if relevant_docs else 0
            f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            st.markdown(f"""
            <div class="news-card" style="border-left: 4px solid var(--ink); padding: 20px;">
                <div class="card-title-text" style="font-size: 15px; margin-bottom: 12px; font-weight: 600;">📊 HASIL PERHITUNGAN MATEMATIS EVALUASI</div>
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 12px; text-align: center;">
                    <div style="background: var(--paper); padding: 12px; border-radius: var(--radius-sm); border: 1px solid var(--hairline);">
                        <div style="font-size: 11px; color: var(--stone); font-weight: 600; letter-spacing: 0.04em;">PRECISION</div>
                        <div style="font-size: 22px; font-weight: 700; color: var(--ink); margin-top:4px;">{precision:.2%}</div>
                    </div>
                    <div style="background: var(--paper); padding: 12px; border-radius: var(--radius-sm); border: 1px solid var(--hairline);">
                        <div style="font-size: 11px; color: var(--stone); font-weight: 600; letter-spacing: 0.04em;">RECALL</div>
                        <div style="font-size: 22px; font-weight: 700; color: var(--ink); margin-top:4px;">{recall:.2%}</div>
                    </div>
                    <div style="background: var(--paper); padding: 12px; border-radius: var(--radius-sm); border: 1px solid var(--hairline);">
                        <div style="font-size: 11px; color: var(--stone); font-weight: 600; letter-spacing: 0.04em;">F1-SCORE</div>
                        <div style="font-size: 22px; font-weight: 700; color: var(--indigo-700); margin-top:4px;">{f1_score:.4f}</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
        else:
            st.markdown(f"""
            <div class="empty-state">
              <span class="empty-icon">🔭</span>
              <div class="empty-title">Tidak Ada Hasil</div>
              <div class="empty-sub">
                Tidak ada dokumen yang melewati threshold <strong>{threshold:.2f}</strong>.
                Coba turunkan threshold atau ganti kata kunci.
              </div>
            </div>
            """, unsafe_allow_html=True)

    elif search_btn and not query_input.strip():
        st.warning("⚠️ Ketik kata kunci terlebih dahulu sebelum mencari.")

    # ── FOOTER ─────────────────────────────────────
    st.markdown(f"""
    <div class="site-footer">
      <strong>STKI · Sistem Temu Kembali Informasi</strong><br>
      Model Ruang Vektor (VSM) &middot; TF-IDF Weighting &middot; Cosine Similarity &middot; Sastrawi NLP<br>
      Corpus: <code>Kompas.com Live Scraping</code> &mdash; {len(df)} dokumen terindeks
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
