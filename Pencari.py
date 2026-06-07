# -*- coding: utf-8 -*-
"""
Mesin Pencari Berita Indonesia
Sistem Temu Kembali Informasi dengan TF-IDF + Query Expansion
Deploy via Streamlit
"""

import re
import time
import pickle
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
# KONFIGURASI HALAMAN
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Mesin Pencari Berita",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────
# CSS CUSTOM (berdasarkan desain HTML yang ada)
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');
@import url('https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/bootstrap-icons.min.css');

html, body, [class*="css"] {
    font-family: 'Poppins', sans-serif;
}

/* ── Background ── */
.stApp {
    background: linear-gradient(135deg, #71d1e4 0%, #4db8d4 50%, #2a9fb8 100%);
    min-height: 100vh;
}

/* ── Sembunyikan elemen bawaan Streamlit ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 2rem; padding-bottom: 2rem; }

/* ── Hero Section ── */
.hero-section {
    text-align:center;
    padding:20px 20px 5px;
}

.hero-logo {
    width: 70px;
    height: 70px;
    margin: 0 auto 20px;
    border-radius: 25px;
    background: rgba(255,255,255,0.2);
    backdrop-filter: blur(10px);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 40px;
    box-shadow: 0 8px 30px rgba(0,0,0,0.2);
    border: 1px solid rgba(255,255,255,0.3);
}

.hero-title {
    font-size: 52px;
    font-weight: 700;
    color: white;
    margin-bottom: 8px;
    text-shadow: 0 2px 15px rgba(0,0,0,0.15);
    letter-spacing: -1px;
}

.hero-subtitle {
    color: rgba(255,255,255,0.9);
    font-size: 17px;
    font-weight: 300;
    margin-bottom: 0;
}

/* Search */
    .search-box{
      position:relative;
      margin-bottom:30px;
    }

    .search-box input{
      width:100%;
      height:70px;
      border:none;
      outline:none;
      border-radius:60px;
      padding:0 75px 0 28px;
      font-size:18px;
      background:rgba(255,255,255,0.18);
      color:white;
      backdrop-filter:blur(10px);
      box-shadow:0 4px 20px rgba(0,0,0,0.1);
    }

    .search-box input::placeholder{
      color:#eee;
    }

    body.dark .search-box input{
      background:rgba(255,255,255,0.08);
    }

    .search-box input:focus{
      transform:scale(1.01);
      box-shadow:0 0 25px rgba(255,255,255,0.2);
    }

    /* Search Button */
    .search-btn{
      position:absolute;
      top:50%;
      right:10px;
      transform:translateY(-50%);
      width:55px;
      height:55px;
      border:none;
      border-radius:50%;
      background:white;
      color:#6a11cb;
      font-size:22px;
      box-shadow:0 5px 20px rgba(255,255,255,0.3);
    }

    body.dark .search-btn{
      background:#111827;
      color:white;
    }

    .search-btn:hover{
      transform:translateY(-50%) scale(1.1);
    }

    /* Responsive */
    @media(max-width:768px){

      .glass-box{
        padding:35px 25px;
      }

      h1{
        font-size:36px;
      }

      .subtitle{
        font-size:15px;
      }

      .search-box input{
        height:60px;
        font-size:16px;
      }

    }

/* ── Selectbox ── */
.stSelectbox > div > div {
    background: rgba(255,255,255,0.18) !important;
    border: 1.5px solid rgba(255,255,255,0.35) !important;
    border-radius: 16px !important;
    color: white !important;
}
.stSelectbox label { color: white !important; font-weight: 500 !important; font-size: 14px !important; }

/* ── Slider ── */
.stSlider label { color: white !important; font-weight: 500 !important; font-size: 14px !important; }
.stSlider p { color: rgba(255,255,255,0.8) !important; }

/* ── Checkbox ── */
.stCheckbox label { color: white !important; font-weight: 400 !important; }

/* ── Hasil pencarian card ── */
.result-card {
    background: rgba(255,255,255,0.92);
    border-radius: 20px;
    padding: 24px 28px;
    margin-bottom: 16px;
    border-left: 4px solid #1a7a8f;
    box-shadow: 0 4px 20px rgba(0,0,0,0.1);
    transition: all 0.25s ease;
    animation: fadeInUp 0.4s ease;
}
.result-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 28px rgba(0,0,0,0.15);
}
.result-rank {
    font-size: 12px;
    font-weight: 600;
    color: #1a7a8f;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 6px;
}
.result-title {
    font-size: 17px;
    font-weight: 600;
    color: #1a2a35;
    margin-bottom: 8px;
    line-height: 1.4;
}
.result-url {
    font-size: 13px;
    color: #1a7a8f;
    margin-bottom: 10px;
    word-break: break-all;
}
.result-score-bar {
    height: 6px;
    background: linear-gradient(90deg, #71d1e4, #1a7a8f);
    border-radius: 10px;
    margin-top: 10px;
}
.result-meta {
    display: flex;
    gap: 12px;
    align-items: center;
    margin-top: 8px;
    flex-wrap: wrap;
}
.badge {
    background: rgba(26,122,143,0.1);
    color: #1a7a8f;
    border-radius: 20px;
    padding: 3px 12px;
    font-size: 12px;
    font-weight: 500;
    border: 1px solid rgba(26,122,143,0.2);
}

/* ── Metrik ── */
.metric-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 16px;
    margin-top: 8px;
}
.metric-box {
    background: rgba(255,255,255,0.15);
    border: 1px solid rgba(255,255,255,0.25);
    border-radius: 16px;
    padding: 20px;
    text-align: center;
    backdrop-filter: blur(10px);
}
.metric-label { color: rgba(255,255,255,0.8); font-size: 13px; font-weight: 400; margin-bottom: 4px; }
.metric-value { color: white; font-size: 28px; font-weight: 700; letter-spacing: -1px; }

/* ── Info banner ── */
.info-banner {
    background: rgba(255,255,255,0.15);
    border: 1px solid rgba(255,255,255,0.25);
    border-radius: 16px;
    padding: 16px 20px;
    color: white;
    font-size: 14px;
    text-align: center;
    margin-bottom: 20px;
}

/* ── Animasi ── */
@keyframes fadeInDown {
    from { opacity: 0; transform: translateY(-30px); }
    to   { opacity: 1; transform: translateY(0); }
}
@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(20px); }
    to   { opacity: 1; transform: translateY(0); }
}

/* ── Footer ── */
.footer {
    text-align: center;
    color: rgba(255,255,255,0.7);
    font-size: 13px;
    padding: 30px 0 10px;
}

/* ── Divider ── */
hr { border-color: rgba(255,255,255,0.2) !important; }

/* ── Spinner ── */
.stSpinner > div { border-top-color: white !important; }

/* ── Expander ── */
.streamlit-expanderHeader {
    color: white !important;
    background: rgba(255,255,255,0.1) !important;
    border-radius: 12px !important;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# INISIALISASI PREPROCESSING TOOLS
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_nlp_tools():
    sw_factory  = StopWordRemoverFactory()
    stopword    = sw_factory.create_stop_word_remover()
    st_factory  = StemmerFactory()
    stemmer     = st_factory.create_stemmer()
    return stopword, stemmer

@st.cache_resource(show_spinner=False)
def load_dataset_and_index():
    """Muat dataset & bangun indeks TF-IDF."""
    URLS = [
        "https://nasional.kompas.com/read/2026/03/09/17303561/prabowo-ingatkan-rakyat-indonesia-bersiap-hadapi-kesulitan-akibat-perang",
        "https://nasional.kompas.com/read/2026/03/10/13125731/memahami-kerja-ombudsman-di-tengah-penggeledahan-terkait-kasus-cpo",
        "https://nasional.kompas.com/read/2026/03/10/12344451/puan-sebut-dpr-dukung-pembatasan-medsos-untuk-anak-agar-tak-kebablasan",
        "https://nasional.kompas.com/read/2026/03/06/20503101/klarifikasi-bahlil-soal-stok-bbm-sisa-20-hari-yang-bikin-warga-panic-buying",
        "https://regional.kompas.com/read/2026/03/06/135609978/jokowi-ungkap-isi-pertemuan-dengan-prabowo-di-istana-bahas-bop-dan-tarif",
        "https://nasional.kompas.com/read/2026/03/09/11502271/kejagung-geledah-kantor-dan-rumah-komisioner-ombudsman-terkait-kasus-cpo",
        "https://nasional.kompas.com/read/2026/03/10/07193531/polri-jelaskan-simulator-berkuda-yang-viral-harganya-rp-1-miliar.",
        "https://nasional.kompas.com/read/2026/03/10/09142961/dua-ott-dalam-seminggu-dua-bupati-ditangkap-kpk.",
        "https://nasional.kompas.com/read/2026/03/10/09411181/kpk-periksa-ketum-pp-japto-soerjosoemarno-jadi-saksi-kasus-gratifikasi.",
        "https://nasional.kompas.com/read/2026/03/06/20471631/semarakkan-ramadhan-kasatgas-tito-salurkan-bantuan-untuk-puluhan-ribu-korban",
        "https://money.kompas.com/read/2026/03/01/055816426/daftar-harga-bbm-terbaru-1-maret-2026-di-seluruh-indonesia",
        "https://money.kompas.com/read/2026/03/07/080016726/cek-harga-pertamax-hari-ini-7-maret-2026-di-spbu-seluruh-indonesia",
        "https://money.kompas.com/read/2026/03/10/204239226/bni-bbni-tebar-dividen-rp-1303-t-analis-soroti-sinyal-stabilitas",
        "https://money.kompas.com/read/2026/03/09/170101426/imbas-fitch-turunkan-outlook-indonesia-investor-mulai-khawatir-kebijakan",
        "https://money.kompas.com/read/2026/03/04/204140126/menkeu-as-tarif-global-15-persen-kemungkinan-berlaku-minggu-ini",
        "https://money.kompas.com/read/2026/03/10/123716426/wifi-dan-nokia-jajaki-pengembangan-infrastruktur-digital-6g",
        "https://money.kompas.com/read/2026/03/07/075736926/harga-emas-di-pegadaian-7-maret-2026-galeri-24-dan-ubs-stabil",
        "https://money.kompas.com/read/2026/03/10/120912026/harga-emas-pegadaian-hari-ini-10-3-kompak-menguat-cek-daftar-ubs-galeri-24",
        "https://money.kompas.com/read/2026/03/04/202228826/selat-hormuz-bergejolak-pengalihan-impor-minyak-ke-as-jadi-alternatif",
        "https://money.kompas.com/read/2026/03/10/040400926/120-juta-unit-sepeda-motor-bensin-dikonversi-ke-listrik-bahlil--untuk-kurangi.",
        "https://tekno.kompas.com/read/2026/03/10/08040087/nintendo-gugat-as-minta-uang-tarif-impor-trump-dikembalikan",
        "https://tekno.kompas.com/read/2026/03/10/07000057/ini-dia-hiroh-phone-hp-yang-punya-tombol-fisik-darurat-untuk-anti-sadap",
        "https://tekno.kompas.com/read/2026/03/09/12240037/daftar-harga-iphone-terbaru-9-maret-2026-iphone-13-iphone-17-series",
        "https://tekno.kompas.com/read/2026/03/09/11220057/8-aplikasi-yang-dilarang-komdigi-buat-diakses-anak-anak-di-bawah-16-tahun",
        "https://tekno.kompas.com/read/2026/03/08/15030037/china-bikin-baterai-nuklir-mini-bisa-tahan-50-tahun-tanpa-dicas.",
        "https://money.kompas.com/read/2026/02/16/162901226/memutus-kutukan-generasi-ketiga-mengapa-hong-kong-jadi-opsi-strategis-bagi",
        "https://tekno.kompas.com/read/2026/03/01/19070037/populasi-robot-ai-diprediksi-meledak-pekerja-manusia-makin-terdesak.",
        "https://tekno.kompas.com/read/2026/03/07/14030087/robot-humanoid-mulai-kerja-di-pabrik-xiaomi.",
        "https://regional.kompas.com/read/2026/03/05/161743778/tni-gerebek-tambang-emas-ilegal-di-madina-6-ekskavator-dan-pekerja",
        "https://surabaya.kompas.com/read/2026/03/10/084602278/kesehatan-menurun-selama-ditahan-terdakwa-pembunuhan-bocah-sd-di-pasuruan",
        "https://surabaya.kompas.com/read/2026/03/10/081939978/pengusaha-parsel-di-surabaya-mengeluh-sepi-dulu-terjual-1000-per-hari-kini",
        "https://regional.kompas.com/read/2026/03/06/134815078/desa-di-kendal-ini-bagikan-thr-ke-warga-setiap-kk-dapat-rp-1-juta",
        "https://regional.kompas.com/read/2026/03/09/193016078/dorong-pad-pemprov-maluku-siapkan-kepulauan-lucipara-jadi-wisata-bahari",
        "https://bandung.kompas.com/read/2026/03/08/115715478/peringatan-nuzulul-quran-2000-al-quran-akan-disalurkan-ke-warga-bogor",
        "https://regional.kompas.com/read/2026/03/10/151404778/napi-lapas-nusakambangan-kabur-hingga-menyeberang-pulau-ujungnya-ditangkap.",
        "https://megapolitan.kompas.com/read/2026/03/09/11560641/cuaca-ekstrem-diprediksi-terjadi-hingga-12-maret-jakarta-masuk-wilayah",
        "https://megapolitan.kompas.com/read/2026/03/09/17075531/detik-detik-warga-periuk-damai-diterjang-banjir-air-datang-tiba-tiba-saat",
        "https://megapolitan.kompas.com/read/2026/03/10/07230411/saat-peringatan-prabowo-soal-sampah-bantargebang-jadi-kenyataan.",
        "https://internasional.kompas.com/read/2026/03/10/061605970/trump-isyaratkan-segera-akhiri-perang-klaim-iran-sudah-lumpuh",
        "https://lestari.kompas.com/read/2026/03/10/081200786/china-targetkan-pangkas-intensitas-karbon-17-persen-pada-2030",
        "https://internasional.kompas.com/read/2026/03/09/080000170/as-disebut-kecewa-israel-gempur-30-pangkalan-minyak-iran-anggap.",
        "https://internasional.kompas.com/read/2026/03/09/050356570/resmi-ayatollah-mojtaba-khamenei-terpilih-jadi-pemimpin-tertinggi-iran.",
        "https://lestari.kompas.com/read/2026/03/09/075648886/klh-tuang-10000-liter-ecoenzym-ke-sungai-cisadane-netralkan-cemaran",
        "https://lestari.kompas.com/read/2026/03/08/150422386/perang-emisi-dan-masa-depan-bumi",
        "https://lestari.kompas.com/read/2026/03/09/161005186/akan-dibor-tahun-ini-pge-siapkan-infrastruktur-pltp-lumut-balai-unit-3",
        "https://lestari.kompas.com/read/2026/03/10/130200086/bab-baru-nilai-ekonomi-karbon",
        "https://lestari.kompas.com/read/2026/03/09/194800786/sering-cek-email-kerja-saat-akhir-pekan-waspada-dampaknya-untuk-mental",
        "https://travel.kompas.com/read/2026/03/09/060600827/7-wisata-sejarah-di-kota-semarang-sudah-ada-yang-dari-abad-ke-17.",
        "https://bandung.kompas.com/read/2026/02/25/204040778/truk-terparkir-seharian-di-bandung-sopir-ternyata-meninggal-di-kabin.",
        "https://travel.kompas.com/read/2026/03/10/050500427/jadwal-dan-harga-tiket-kereta-wisata-ambarawa-saat-libur-lebaran-18-31-maret.",
    ]
    return URLS

# ─────────────────────────────────────────────
# FUNGSI SCRAPING
# ─────────────────────────────────────────────
@st.cache_data(show_spinner=False, ttl=3600)
def scrape_articles(urls):
    headers = {"User-Agent": "Mozilla/5.0"}
    data = []
    for i, url in enumerate(urls):
        try:
            page    = requests.get(url.strip(), headers=headers, timeout=10)
            soup    = BeautifulSoup(page.text, "html.parser")
            h1      = soup.find("h1")
            judul   = h1.get_text(strip=True) if h1 else "Tanpa Judul"
            content = soup.find("div", {"class": "read__content"})
            text    = ""
            if content:
                paragraf = content.find_all("p")
                text = " ".join([p.get_text() for p in paragraf])
            data.append({"No": i + 1, "URL": url.strip(), "Judul": judul, "Text": text})
        except Exception:
            data.append({"No": i + 1, "URL": url.strip(), "Judul": "Error", "Text": ""})
    return pd.DataFrame(data)

# ─────────────────────────────────────────────
# FUNGSI PREPROCESSING
# ─────────────────────────────────────────────
def preprocess_text(text, stopword, stemmer):
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = stopword.remove(text)
    text = stemmer.stem(text)
    return text

def preprocess_query(query_text, stopword, stemmer):
    q = query_text.lower()
    q = re.sub(r'[^a-zA-Z\s]', '', q)
    q = stopword.remove(q)
    tokens = q.split()
    tokens = [stemmer.stem(w) for w in tokens]
    return tokens

# ─────────────────────────────────────────────
# FUNGSI SINONIM (Query Expansion)
# ─────────────────────────────────────────────
thesaurus_cache = {}

def get_sinonim(kata):
    if kata in thesaurus_cache:
        return thesaurus_cache[kata]
    try:
        data_form = {"q": kata}
        encoded   = urllib.parse.urlencode(data_form).encode("utf-8")
        content   = urllib.request.urlopen(
            "http://www.sinonimkata.com/search.php", encoded, timeout=5
        )
        soup      = BeautifulSoup(content, 'html.parser')
        td        = soup.find('td', attrs={'width': '90%'})
        if td:
            synonym = [a.getText() for a in td.find_all('a')]
            result  = [kata] + synonym
        else:
            result  = [kata]
    except Exception:
        result = [kata]
    thesaurus_cache[kata] = result
    return result

def filter_sinonim_dinamis(kata, sinonim_list, query_words, vectorizer,
                           tfidf_matrix, top_n=3, threshold=0.7):
    query_asli  = " ".join(query_words)
    q_asli_vec  = vectorizer.transform([query_asli])
    result_asli = cosine_similarity(tfidf_matrix, q_asli_vec).flatten()
    hasil       = []
    idx_kata    = query_words.index(kata)

    for s in sinonim_list:
        if s == kata:
            hasil.append((s, 1.0))
            continue
        query_variasi       = query_words.copy()
        query_variasi[idx_kata] = s
        q_var_vec           = vectorizer.transform([" ".join(query_variasi)])
        result_variasi      = cosine_similarity(tfidf_matrix, q_var_vec).flatten()
        norm_asli           = np.linalg.norm(result_asli)
        norm_var            = np.linalg.norm(result_variasi)
        if norm_asli > 0 and norm_var > 0:
            skor = np.dot(result_asli, result_variasi) / (norm_asli * norm_var)
        else:
            skor = 0.0
        if skor >= threshold:
            hasil.append((s, skor))

    hasil = sorted(hasil, key=lambda x: x[1], reverse=True)
    return [x[0] for x in hasil[:top_n]]

# ─────────────────────────────────────────────
# FUNGSI PENCARIAN UTAMA
# ─────────────────────────────────────────────
def search(query_tokens, vectorizer, tfidf_matrix, df,
           use_expansion=False, threshold=0.1, top_n=10):
    if use_expansion:
        list_synonym = []
        for kata in query_tokens:
            sinonim = get_sinonim(kata)
            sinonim = filter_sinonim_dinamis(
                kata, sinonim, query_tokens, vectorizer, tfidf_matrix,
                top_n=3, threshold=0.7
            )
            list_synonym.append(sinonim)

        qs = set()
        for i, kata in enumerate(query_tokens):
            for s in list_synonym[i]:
                kombinasi    = query_tokens.copy()
                kombinasi[i] = s
                qs.add(" ".join(kombinasi))
        qs.add(" ".join(query_tokens))
    else:
        qs    = {" ".join(query_tokens)}
        list_synonym = [[k] for k in query_tokens]

    max_result = []
    for q in qs:
        q_vec  = vectorizer.transform([q])
        result = cosine_similarity(tfidf_matrix, q_vec)
        for i in range(len(result)):
            if result[i][0] > threshold:
                max_result.append([i, result[i][0], q])

    max_result = sorted(max_result, key=lambda x: x[1], reverse=True)
    seen, new_result = set(), []
    for item in max_result:
        if item[0] not in seen:
            seen.add(item[0])
            new_result.append(item)

    # Evaluasi P/R/F1
    q_vec_asli  = vectorizer.transform([" ".join(query_tokens)])
    relevant    = {i for i in range(len(df))
                   if cosine_similarity(tfidf_matrix[i], q_vec_asli)[0][0] >= threshold}
    retrieved   = {item[0] for item in new_result}
    tp          = relevant & retrieved
    precision   = len(tp) / len(retrieved) if retrieved else 0
    recall      = len(tp) / len(relevant)  if relevant  else 0
    f1          = (2 * precision * recall) / (precision + recall) \
                  if (precision + recall) > 0 else 0

    synonyms_used = list_synonym if use_expansion else None
    return new_result[:top_n], precision, recall, f1, synonyms_used

# ─────────────────────────────────────────────
# UI UTAMA
# ─────────────────────────────────────────────
def main():
    stopword, stemmer = load_nlp_tools()
    URLS              = load_dataset_and_index()

    # ── Hero ──
    st.markdown("""
    <div class="hero-section">
        <div class="hero-logo">🔍</div>
        <h1 class="hero-title">Search Engine</h1>
        <p class="hero-subtitle">Mesin Pencari Berita Indonesia</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Panel Pencarian ──
    use_expansion = True
    threshold = 0.1
    top_n = 10
    
    query_input = st.text_input("Masukkan kata kunci",placeholder="Contoh: harga bbm, AI, Prabowo...")
        use_expansion = True
        
        col1, col2 = st.columns([2,1])
        
        with col1:
            threshold = st.slider(
                "Threshold relevansi",
                0.01,
                0.5,
                0.1,
                0.01
            )
        
        with col2:
            top_n = st.selectbox(
                "Jumlah hasil",
                [5,10,20],
                index=1
            )
        
        search_btn = st.button(
            "🔍 Cari Berita",
            use_container_width=True
        )
        st.markdown('</div>', unsafe_allow_html=True)  

    # ── Load & Index data ──
    if "df" not in st.session_state or "vectorizer" not in st.session_state:
        with st.spinner("⏳ Memuat & mengindeks dataset berita..."):
            df        = scrape_articles(tuple(URLS))
            df["clean_text"] = df["Text"].apply(
                lambda x: preprocess_text(x, stopword, stemmer)
            )
            vectorizer   = TfidfVectorizer(use_idf=True)
            tfidf_matrix = vectorizer.fit_transform(df["clean_text"].tolist())
            st.session_state["df"]           = df
            st.session_state["vectorizer"]   = vectorizer
            st.session_state["tfidf_matrix"] = tfidf_matrix

    df           = st.session_state["df"]
    vectorizer   = st.session_state["vectorizer"]
    tfidf_matrix = st.session_state["tfidf_matrix"]

    col_left2, col_center2, col_right2 = st.columns([1, 3, 1])
    with col_center2:
        st.markdown(f"""
        <div class="info-banner">
            📰 <b>{len(df)}</b> artikel berita terindeks · Siap dicari
        </div>
        """, unsafe_allow_html=True)

    # ── Proses pencarian ──
    if search_btn and query_input.strip():
        query_tokens = preprocess_query(query_input, stopword, stemmer)

        if not query_tokens:
            st.warning("Query tidak menghasilkan token yang valid. Coba kata lain.")
            return

        with st.spinner("🔍 Memproses pencarian..."):
            results, precision, recall, f1, synonyms_used = search(
                query_tokens, vectorizer, tfidf_matrix, df,
                use_expansion=use_expansion,
                threshold=threshold,
                top_n=top_n,
            )

        col_left3, col_center3, col_right3 = st.columns([1, 3, 1])
        with col_center3:
            # Metrik evaluasi
            st.markdown(f"""
            <div class="glass-card">
                <div style="color:white;font-weight:600;font-size:15px;margin-bottom:14px;">
                    📊 Evaluasi Hasil — query: <i>"{query_input}"</i>
                    → token: <code style="background:rgba(255,255,255,0.2);padding:2px 8px;border-radius:8px;">
                    {" ".join(query_tokens)}</code>
                </div>
                <div class="metric-grid">
                    <div class="metric-box">
                        <div class="metric-label">Precision</div>
                        <div class="metric-value">{precision:.2f}</div>
                    </div>
                    <div class="metric-box">
                        <div class="metric-label">Recall</div>
                        <div class="metric-value">{recall:.2f}</div>
                    </div>
                    <div class="metric-box">
                        <div class="metric-label">F1-Score</div>
                        <div class="metric-value">{f1:.2f}</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Sinonim yang dipakai
            if usynonyms_used:
                syn_html = ""
                for tok, syns in zip(query_tokens, synonyms_used):
                    badges = "".join(f'<span class="badge">{s}</span> ' for s in syns)
                    syn_html += f'<div style="margin-bottom:6px;color:rgba(255,255,255,0.85);font-size:13px;"><b>{tok}</b> → {badges}</div>'
                st.markdown(f"""
                <div class="glass-card">
                    <div style="color:white;font-weight:600;font-size:14px;margin-bottom:10px;">🔄 Sinonim yang digunakan</div>
                    {syn_html}
                </div>
                """, unsafe_allow_html=True)

            # Daftar hasil
            if results:
                st.markdown(f"""
                <div style="color:white;font-weight:600;font-size:16px;margin:10px 0 14px;">
                    📋 {len(results)} Berita Ditemukan
                </div>
                """, unsafe_allow_html=True)

                for rank, (idx, score, q_used) in enumerate(results, 1):
                    bar_width = min(int(score * 100 / max(r[1] for r in results) * 100), 100)
                    judul = df["Judul"].iloc[idx]
                    url   = df["URL"].iloc[idx]
                    domain = url.split("/")[2] if "/" in url else url

                    st.markdown(f"""
                    <div class="result-card">
                        <div class="result-rank">#{rank} · Relevansi {score:.4f}</div>
                        <div class="result-title">{judul}</div>
                        <div class="result-url">🔗 {domain}</div>
                        <div class="result-meta">
                            <span class="badge">Score: {score:.4f}</span>
                            <span class="badge">Query: {q_used}</span>
                            <a href="{url}" target="_blank"
                               style="color:#1a7a8f;font-size:13px;font-weight:500;text-decoration:none;">
                               Buka artikel ↗
                            </a>
                        </div>
                        <div class="result-score-bar" style="width:{bar_width}%"></div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="glass-card" style="text-align:center;color:white;">
                    😕 Tidak ada hasil ditemukan dengan threshold ini.<br>
                    <small>Coba turunkan nilai threshold atau gunakan query yang berbeda.</small>
                </div>
                """, unsafe_allow_html=True)

    elif search_btn and not query_input.strip():
        col_left4, col_center4, col_right4 = st.columns([1, 3, 1])
        with col_center4:
            st.markdown("""
            <div class="glass-card" style="text-align:center;color:white;">
                ⚠️ Masukkan kata kunci terlebih dahulu.
            </div>
            """, unsafe_allow_html=True)

    # ── Footer ──
    st.markdown("""
    <div class="footer">
        Sistem Temu Kembali Informasi · TF-IDF + Cosine Similarity + Query Expansion<br>
        Sumber data: Kompas.com · Dibangun dengan Streamlit
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
