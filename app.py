# app.py
import os
import re
import unicodedata
from datetime import date as dt_date

import streamlit as st
import joblib
import numpy as np
import pandas as pd
import holidays


# =========================
# KONFİG
# =========================
st.set_page_config(page_title="İBB Raylı Sistem Tahmin (RF+CatBoost)", layout="wide")
st.title("🚇 İBB Raylı Sistem Tahmin • RF(0.7) + CatBoost(0.3)")

BUNDLE_PATH = "bundle_rf_catboost.joblib"  # aynı klasörde
DEFAULT_ALPHA = 0.7


# =========================
# 1) İSTASYON-İLÇE LİSTESİ (SENİN VERDİĞİN RAW)
#    -> Dropdown için burada parse ediyoruz.
# =========================
STATION_DISTRICT_RAW = r"""
4 Levent 2 Güney     Kağıthane
4 Levent Kuzey     Kağıthane
AKSARAY         Fatih
AKSARAY 1         Fatih
ALTINŞEHİR      Ümraniye
ALTUNİZADE 1       Üsküdar
ALTUNİZADE 2       Üsküdar
ALİBEYKÖY BATI    Eyüpsultan
ALİBEYKÖY DOĞU    Eyüpsultan
ATAKOY      Bakırköy
Acýbadem (Batý)       Kadıköy
Acýbadem (Doğu)       Kadıköy
Acıbadem (Batı)       Kadıköy
Acıbadem (Doğu)       Kadıköy
Aksaray         Fatih
Aksaray 1         Fatih
Akýncýlar      Güngören
Akıncılar      Güngören
Akşemsettin   Zeytinburnu
Ali Fuat Başgil Gaziosmanpaşa
Alibeyköy         Fatih
Alibeyköy Metro         Fatih
Altunizade 1       Üsküdar
Altunizade 2       Üsküdar
Altınşehir      Ümraniye
Ataköy      Bakırköy
Atalar        Kartal
Atatürk Oto Sanayi Güney         Şişli
Atatürk Oto Sanayi Kuzey         Şişli
Atatürk Öğrenci Yurdu   Zeytinburnu
Aydýntepe         Tuzla
Aydıntepe         Tuzla
Ayrýlýkçeşme       Kadıköy
Ayrýlýkçeşmesi       Kadıköy
Ayrılıkçeşme       Kadıköy
Ayrılıkçeşmesi       Kadıköy
Ayvansaray         Fatih
BAGCILAR MEYDAN      Bağcılar
BAHCELIEVLER      Bakırköy
BAKIRKOY      Bakırköy
BAYRAMPASA    Eyüpsultan
BAĞLARBAŞI       Üsküdar
BULGURLU       Üsküdar
Bahçelievler      Bakırköy
Bakýrköy-1      Bakırköy
Bakýrköy-2      Bakırköy
Bakırköy      Bakırköy
Bakırköy İdo      Bakırköy
Bakırköy-1      Bakırköy
Bakırköy-2      Bakırköy
Balat         Fatih
Bayrampaşa    Eyüpsultan
Bağcýlar      Bağcılar
Bağcılar      Bağcılar
Bağcılar Meydan      Bağcılar
Bağlarbaşý       Üsküdar
Bağlarbaşı       Üsküdar
Başak        Kartal
Başak Konutlarý    Başakşehir
Başak Konutları    Başakşehir
Baştabya    Bayrampaşa
Bereç Gaziosmanpaşa
Beyazýt         Fatih
Beyazıt         Fatih
Beyoğlu       Beyoğlu
Bostancý       Kadıköy
Bostancý (Batý)       Kadıköy
Bostancý (Doğu)       Kadıköy
Bostancý-1       Kadıköy
Bostancý-2       Kadıköy
Bostancı       Kadıköy
Bostancı (Batı)       Kadıköy
Bostancı (Doğu)       Kadıköy
Bostancı-1       Kadıköy
Bostancı-2       Kadıköy
Boğaz Köprüsü 2       Üsküdar
Boğaziçi       Sarıyer
Bulgurlu       Üsküdar
Cami      Güngören
Cebeci Gaziosmanpaşa
Cep Otogar         Fatih
Cevizli-1        Kartal
Cevizli-2        Kartal
Cibali         Fatih
Cumhuriyet    Bayrampaşa
DAVUTPASA      Güngören
DUDULLU      Ümraniye
Darüşşafaka       Sarıyer
Darýca         Tuzla
Darıca         Tuzla
Davutpaşa      Güngören
Demirkapý    Eyüpsultan
Demirkapı    Eyüpsultan
Dudullu      Ümraniye
EMNIYET         Fatih
ESENLER    Bayrampaşa
Edirnekapý    Eyüpsultan
Edirnekapı    Eyüpsultan
Eminönü         Fatih
Eminönü 2         Fatih
Emniyet         Fatih
Erenköy       Kadıköy
Esenkent Cevizli       Maltepe
Esenler    Bayrampaşa
Etiler         Şişli
Eyüp    Eyüpsultan
Eyüp Devlet Hastanesi         Fatih
Eyüp Teleferik         Fatih
FEVZİ ÇAKMAK        Pendik
FISTIKAĞACI       Üsküdar
Fatih         Tuzla
Fener         Fatih
Feneryolu       Kadıköy
Feshane         Fatih
Fetihkapý   Zeytinburnu
Fetihkapı   Zeytinburnu
Fevzi Çakmak        Pendik
Florya      Bakırköy
Florya aqua      Bakırköy
Fýndýklý       Beyoğlu
Fýndýkzade         Fatih
Fýstýkağacý       Üsküdar
Fındıklı       Beyoğlu
Fındıkzade         Fatih
Fıstıkağacı       Üsküdar
Gayrettepe         Şişli
Gebze-1         Tuzla
Gebze-2         Tuzla
GÖZTEPE BATI       Kadıköy
GÖZTEPE DOĞU       Kadıköy
Göztepe       Kadıköy
Göztepe       Üsküdar
Gülhane         Fatih
Gülsuyu       Maltepe
Güneştepe      Güngören
Güngören      Güngören
Güzelyalý        Pendik
Güzelyalı        Pendik
HAVAALANI      Bakırköy
Hacýosman       Sarıyer
Hacýşükrü Gaziosmanpaşa
Hacı Şükrü Gaziosmanpaşa
Hacıosman       Sarıyer
Haliç güney         Fatih
Haliç kuzey         Fatih
Halkalý      Bakırköy
Halkalı      Bakırköy
Haseki         Fatih
Hastane (Batý)        Kartal
Hastane (Batı)        Kartal
Hastane (Doğu/Adliye)        Kartal
Havaalanı      Bakırköy
Haznedar      Bağcılar
Huzurevi       Maltepe
IDTM      Bakırköy
IHLAMUR KUYU      Ümraniye
Ihlamurkuyu      Ümraniye
KABATAS       Beyoğlu
KARADENİZ MAH. BATI Gaziosmanpaşa
KARADENİZ MAH. DOĞU Gaziosmanpaşa
KARTALTEPE Gaziosmanpaşa
KAZIMKARABEKİR Gaziosmanpaşa
KAĞITHANE BATI     Kağıthane
KAĞITHANE DOGU     Kağıthane
KIRAZLI      Bağcılar
KISIKLI       Üsküdar
Kabataş       Beyoğlu
Kabataş 2       Beyoğlu
Kadýköy (Batý)       Kadıköy
Kadýköy (Doğu)       Kadıköy
Kadýköy Çayýrbaşý       Kadıköy
Kadıköy (Batı)       Kadıköy
Kadıköy (Doğu)       Kadıköy
Karadeniz Mahallesi Gaziosmanpaşa
Karaköy       Beyoğlu
Kartal        Kartal
Kartal (Batý)        Kartal
Kartal (Batı)        Kartal
Kartal (Doğu)        Kartal
Kartaltepe Gaziosmanpaşa
Kayaşehir Merkez      Bağcılar
Kaynarca        Pendik
Kazlýçeşme   Zeytinburnu
Kazlıçeşme   Zeytinburnu
Keresteciler      Güngören
Kiptaş Venezia Gaziosmanpaşa
Kirazlý      Bağcılar
Kirazlı      Bağcılar
Kozyatağý       Kadıköy
Kozyatağı       Kadıköy
Kurtköy        Pendik
Küçükpazar         Fatih
Küçükyalý       Maltepe
Küçükyalý-1       Maltepe
Küçükyalý-2       Maltepe
Küçükyalı       Maltepe
Küçükyalı-1       Maltepe
Küçükyalı-2       Maltepe
Küçükçekmece      Bakırköy
Kýsýklý       Üsküdar
Kısıklı       Üsküdar
Laleli         Fatih
Levent 2 Kuzey         Şişli
Levent Batý konkors         Şişli
Levent Batı konkors         Şişli
Levent Doğu konkors         Şişli
Levent Güney         Şişli
M.kemal      Bakırköy
M2 Gayrettepe         Şişli
M4 KURTKÖY        Pendik
M7 FULYA         Şişli
M7 YILDIZ 1         Şişli
M7 YILDIZ 2         Şişli
MAHMUTBEY M3 HOL 3      Bağcılar
MAHMUTBEY M3 HOL 4      Bağcılar
MAHMUTBEY M7 HOL 1      Bağcılar
MAHMUTBEY M7 HOL 2       Avcılar
MECİDİYEKÖY BATI         Şişli
MECİDİYEKÖY DOĞU         Şişli
MENDERES    Bayrampaşa
MERTER      Güngören
Mahmutbey      Bağcılar
Mahmutbey M7 Hol 1      Bağcılar
Mahmutbey M7 Hol 2      Güngören
Mahmutbey M7 Hol 3      Güngören
Mahmutbey M7 Hol 4      Güngören
Maltepe       Maltepe
Maçka         Şişli
Meclis      Ümraniye
MehmetAkif      Güngören
Menderes    Bayrampaşa
Merkezefendi   Zeytinburnu
Merter      Güngören
Mescidi Selam Gaziosmanpaşa
Metris Gaziosmanpaşa
Metrokent    Başakşehir
Mithatpaşa   Zeytinburnu
Molla Gürani      Bağcılar
NECİP FAZIL      Ümraniye
NURTEPE BATI     Kağıthane
NURTEPE DOĞU     Kağıthane
Necip Fazıl      Ümraniye
Nispetiye         Şişli
ORUÇREİS BATI      Bağcılar
ORUÇREİS DOĞU      Bağcılar
OTOGAR Gaziosmanpaşa
OTOGAR 1 Gaziosmanpaşa
Onurkent    Başakşehir
Osmanbey 2 Güney         Şişli
Osmanbey Kuzey         Şişli
Osmangazi         Tuzla
Otogar Gaziosmanpaşa
Otogar 1 Gaziosmanpaşa
Pazartekke         Fatih
Pendik        Pendik
Pendik (Batý)        Pendik
Pendik (Batı)        Pendik
Pendik (Doğu)        Pendik
Pierloti    Eyüpsultan
Rami    Eyüpsultan
SABIHA GOKCEN        Pendik
Sabiha Gökçen Havalimanı        Pendik
Samandıra Merkez      Ümraniye
Sanayi Mah. Güney       Sarıyer
Sanayi Mah. Kuzey       Sarıyer
Sancaktepe      Ümraniye
Sarıgazı      Ümraniye
Sağmalcılar Gaziosmanpaşa
Seyrantepe 1 Batı       Sarıyer
Seyrantepe 2 Doğu       Sarıyer
Seyrantepe 3 Stad Girişi       Sarıyer
Silahtarağa         Fatih
Sirkeci         Fatih
Sirkeci-1         Fatih
Sirkeci-2         Fatih
Sirkeci-3         Fatih
Sirkeci-4         Fatih
Siteler    Başakşehir
Soğanlı      Güngören
Soğanlık        Kartal
Suadiye       Kadıköy
Sultanahmet         Fatih
Söğütlüçeşme       Kadıköy
Süreyya plajı       Maltepe
TAKSIM       Beyoğlu
TEKSTİLKENT    Bayrampaşa
TERAZIDERE    Bayrampaşa
Taksim       Beyoğlu
Taksim Güney       Beyoğlu
Tavşantepe (Batı)        Pendik
Tavşantepe (Doğu)        Pendik
Taşköprü Gaziosmanpaşa
Terazidere    Bayrampaşa
Tersane-1        Pendik
Tersane-2        Pendik
Tophane       Beyoğlu
Topkapı   Zeytinburnu
Toplu Konutlar    Başakşehir
Topçular    Eyüpsultan
Turgut Özal    Başakşehir
Tuzla         Tuzla
UCYUZLU      Bağcılar
ULUBATLI         Fatih
Ulubatlı         Fatih
Universite         Fatih
Vatan    Eyüpsultan
Vezneciler Güney         Fatih
Vezneciler Kuzey         Fatih
YAMANEVLER      Ümraniye
YAYALAR        Pendik
YENIBOSNA      Bakırköy
YENIKAPI         Fatih
YENİMAHALLE      Bağcılar
YEŞİLPINAR    Eyüpsultan
Yakacık (Batı)        Kartal
Yakacık (Doğu)        Kartal
Yamanevler      Ümraniye
Yayalar        Pendik
Yeni Mahalle      Bağcılar
Yenibosna      Bakırköy
Yenikapı Güney         Fatih
Yenikapı Kuzey         Fatih
Yenikapı-1         Fatih
Yenikapı-2         Fatih
Yenikapı-3         Fatih
Yenisahra       Kadıköy
Yeşilköy      Bakırköy
Yeşilyurt      Bakırköy
Yunus        Kartal
Yusufpaşa         Fatih
ZEYTINBURNU      Bakırköy
Zeytinburnu   Zeytinburnu
Zeytinburnu 2      Bakırköy
ÇAKMAK      Ümraniye
ÇARŞI      Ümraniye
ÇAĞLAYAN BATI     Kağıthane
ÇAĞLAYAN DOĞU     Kağıthane
ÇEKMEKÖY 1      Ümraniye
ÇEKMEKÖY 2      Ümraniye
ÇIRÇIR BATI    Eyüpsultan
ÇIRÇIR DOĞU    Eyüpsultan
Çakmak      Ümraniye
Çapa         Fatih
Çarşı      Ümraniye
Çayırova         Tuzla
Çemberlitaş         Fatih
Özgürlük Meydanı Güney      Bakırköy
ÜSKÜDAR 1       Üsküdar
ÜSKÜDAR 2       Üsküdar
Ümraniye       Üsküdar
Ünalan       Üsküdar
Üsküdar 1       Üsküdar
Üsküdar 2       Üsküdar
İTÜ Güney         Şişli
İTÜ kuzey         Şişli
İdealtepe       Maltepe
İkitelli Sanayi    Başakşehir
İmam Hatip Lisesi      Ümraniye
İncirli      Bakırköy
İstoç      Bağcılar
İçmeler         Tuzla
Şehir Hastanesi    Başakşehir
Şehitlik    Eyüpsultan
Şişhane Güney       Beyoğlu
Şişhane Kuzey       Beyoğlu
Şişli 2 Kuzey         Şişli
Şişli Güney         Şişli
"""


def fix_weird_tr_chars(s: str) -> str:
    # Sık görülen encoding bozukluklarını düzelt
    repl = {
        "ý": "ı", "Ý": "İ",
        "þ": "ş", "Þ": "Ş",
        "ð": "ğ", "Ð": "Ğ",
        "Þ": "Ş", "þ": "ş",
        "Â": "",  "á": "a", "Á": "A",
    }
    for k, v in repl.items():
        s = s.replace(k, v)
    return s


def normalize_space(s: str) -> str:
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    return s


def slugify_tr(s: str) -> str:
    s = fix_weird_tr_chars(s)
    s = s.strip().lower()
    tr_map = str.maketrans({
        "ç": "c", "ğ": "g", "ı": "i", "ö": "o", "ş": "s", "ü": "u",
        "Ç": "c", "Ğ": "g", "İ": "i", "Ö": "o", "Ş": "s", "Ü": "u",
    })
    s = s.translate(tr_map)
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = re.sub(r"[^a-z0-9]+", "_", s).strip("_")
    return s


def parse_station_district(raw: str):
    pairs = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        line = fix_weird_tr_chars(line)
        # 2+ boşluk / tab ile ayır (istasyon adı içinde tek boşluk olabilir)
        parts = re.split(r"\s{2,}|\t+", line)
        if len(parts) < 2:
            # olmadıysa son boşluktan ayırmayı dene (çok nadir)
            m = re.match(r"^(.*)\s+([A-Za-zÇĞİÖŞÜçğıöşü]+)$", line)
            if not m:
                continue
            station = m.group(1)
            district = m.group(2)
        else:
            station, district = parts[0], parts[1]

        station = normalize_space(station)
        district = normalize_space(district)

        if station and district:
            pairs.append((station, district))

    # Aynı (station,district) tekrarlarını temizle
    uniq = []
    seen = set()
    for s, d in pairs:
        key = (s, d)
        if key not in seen:
            seen.add(key)
            uniq.append((s, d))
    return uniq


STATION_DISTRICT_PAIRS = parse_station_district(STATION_DISTRICT_RAW)

# Dropdown için benzersiz label (aynı istasyon farklı ilçe çıkabilir -> label’e ilçe ekliyoruz)
OPTION_LABELS = [f"{s} — {d}" for s, d in STATION_DISTRICT_PAIRS]
LABEL_TO_PAIR = {f"{s} — {d}": (s, d) for s, d in STATION_DISTRICT_PAIRS}


# =========================
# 2) MEB OKUL TAKVİMİ (2022–2024) + 2024 sonu için 2024-2025 1. dönem
#    Kaynak mantığı:
#    - 2022-2023: 12.09.2022–16.06.2023, ara tatiller 14-18 Kas 2022, 23 Oca–3 Şub 2023, 17–20 Nis 2023
#    - 2023-2024: 11.09.2023–14.06.2024, ara tatil 13-17 Kas 2023, yarıyıl 22 Oca–2 Şub 2024, ara tatil 8-12 Nis 2024
#    - 2024-2025 (2024 kısmı için): dönem başlangıcı 09.09.2024, ara tatil 11-15 Kas 2024, dönem 17.01.2025’e kadar
# =========================
def in_any_range(d: dt_date, ranges):
    for a, b in ranges:
        if a <= d <= b:
            return True
    return False


# Dönem aralıkları (okul açık olabileceği geniş çerçeve)
SCHOOL_TERMS = [
    (dt_date(2022, 9, 12), dt_date(2023, 6, 16)),
    (dt_date(2023, 9, 11), dt_date(2024, 6, 14)),
    (dt_date(2024, 9, 9),  dt_date(2025, 1, 17)),  # 2024 sonunu kapsasın diye
]

# Tatil/break aralıkları
SCHOOL_BREAKS = [
    (dt_date(2022, 11, 14), dt_date(2022, 11, 18)),
    (dt_date(2023, 1, 23),  dt_date(2023, 2, 3)),
    (dt_date(2023, 4, 17),  dt_date(2023, 4, 20)),

    (dt_date(2023, 11, 13), dt_date(2023, 11, 17)),
    (dt_date(2024, 1, 22),  dt_date(2024, 2, 2)),
    (dt_date(2024, 4, 8),   dt_date(2024, 4, 12)),

    (dt_date(2024, 11, 11), dt_date(2024, 11, 15)),
]


@st.cache_resource
def tr_holidays():
    return holidays.Turkey()


TR_HOLIDAYS = tr_holidays()


def compute_calendar_features(d: dt_date):
    weekday_num = d.weekday()  # Mon=0..Sun=6
    is_weekend = int(weekday_num >= 5)
    is_weekday = int(not is_weekend)

    year, month, day = d.year, d.month, d.day
    weekofyear = int(d.isocalendar().week)
    quarter = (month - 1) // 3 + 1

    is_official_holiday = int(d in TR_HOLIDAYS)
    is_holiday = int(is_official_holiday == 1)  # veri setindeki mantığa uyum

    # MEB okul günü:
    # - ilgili dönemin içinde mi?
    # - hafta sonu değil
    # - resmi tatil değil
    # - ara/yarıyıl tatil aralığında değil
    in_term = in_any_range(d, SCHOOL_TERMS)
    in_break = in_any_range(d, SCHOOL_BREAKS)
    is_school_day = int(in_term and (not is_weekend) and (not is_official_holiday) and (not in_break))

    # Senin kolonların:
    Hafta_Sonu = int(is_weekend)
    Tatiller = int(is_official_holiday)
    Okul_Gunleri = int(is_school_day)

    return {
        "year": year,
        "month": month,
        "day": day,
        "weekday_num": weekday_num,
        "weekofyear": weekofyear,
        "quarter": quarter,
        "is_weekday": is_weekday,
        "is_weekend": is_weekend,
        "is_official_holiday": is_official_holiday,
        "is_holiday": is_holiday,
        "is_school_day": is_school_day,
        "Hafta Sonu": Hafta_Sonu,
        "Tatiller": Tatiller,
        "Okul Günleri": Okul_Gunleri,
    }


# =========================
# 3) MODEL YÜKLE
# =========================
@st.cache_resource
def load_bundle(path: str):
    return joblib.load(path)


if not os.path.exists(BUNDLE_PATH):
    st.error(f"❌ `{BUNDLE_PATH}` bulunamadı. Dosya app.py ile aynı klasörde olmalı.")
    st.stop()

bundle = load_bundle(BUNDLE_PATH)
rf_pipe = bundle.get("rf_pipe")
cat_pipe = bundle.get("cat_pipe")
alpha = float(bundle.get("alpha", DEFAULT_ALPHA))

if rf_pipe is None or cat_pipe is None:
    st.error("❌ Bundle içinde `rf_pipe` veya `cat_pipe` yok. Bundle yapısını kontrol et.")
    st.stop()

st.caption(f"Ağırlıklar: **{alpha:.2f} RF** + **{1-alpha:.2f} CatBoost**")


# =========================
# 4) INPUT UI (kullanıcıdan istenen az şey)
# =========================
with st.sidebar:
    st.header("🧾 Girdiler")

    d = st.date_input("Tarih", value=dt_date(2024, 12, 1))
    choice = st.selectbox("İstasyon", options=OPTION_LABELS)

    sunshine_hours = st.number_input("Güneşlenme (saat) • sunshine_hours", value=0.0, step=0.1)
    rain_mm = st.number_input("Yağış (mm) • rain_mm", value=0.0, step=0.1)
    tmax_c = st.number_input("Maks. Sıcaklık (°C) • tmax_c", value=20.0, step=0.1)
    tmin_c = st.number_input("Min. Sıcaklık (°C) • tmin_c", value=10.0, step=0.1)
    passage_cnt = st.number_input("passage_cnt", value=0.0, step=1.0)

station_name, district_name = LABEL_TO_PAIR[choice]
district_norm = slugify_tr(district_name)

cal = compute_calendar_features(d)

# =========================
# 5) FEATURE BUILDER (eksik kolonları otomatik tamamlar)
# =========================
def infer_required_columns(pipe):
    # Pipeline/estimator hangi kolonları bekliyor? Bulabilirsek otomatikleşir.
    req = getattr(pipe, "feature_names_in_", None)
    if req is not None:
        return list(req)

    # Bazı durumlarda preprocessor içinde tutulur
    try:
        for name, step in getattr(pipe, "named_steps", {}).items():
            req2 = getattr(step, "feature_names_in_", None)
            if req2 is not None:
                return list(req2)
    except Exception:
        pass

    # fallback: bizim bildiğimiz temel kolon seti
    return []


def build_X():
    date_str = d.strftime("%Y-%m-%d")

    # Kullanıcıdan gelen minimal hava -> türetmeler
    tmean_c = (float(tmax_c) + float(tmin_c)) / 2.0
    sunshine_sec = float(sunshine_hours) * 3600.0

    # “Model isterse lazım olur” diye otomatik doldurduklarımız
    base = {
        "station_name": station_name,
        "district_name": district_name,
        "district_norm": district_norm,
        "date": date_str,

        "passage_cnt": float(passage_cnt),

        # kullanıcıdan
        "sunshine_hours": float(sunshine_hours),
        "rain_mm": float(rain_mm),
        "tmax_c": float(tmax_c),
        "tmin_c": float(tmin_c),

        # türetilen
        "tmean_c": float(tmean_c),
        "sunshine_sec": float(sunshine_sec),

        # genelde rain ile aynı tutulur
        "precip_mm": float(rain_mm),

        # hissedilen sıcaklıkları basit eşle (API yoksa en makul yaklaşım)
        "tapp_max_c": float(tmax_c),
        "tapp_min_c": float(tmin_c),
        "tapp_mean_c": float(tmean_c),

        # kar vb yoksa 0
        "snowfall_cm": 0.0,
        "snow_depth_cm": 0.0,
        "et0_mm": 0.0,

        # sabit varsayımlar (istersen sonra gerçek API ile doldururuz)
        "wind10m_mean_kmh": 10.0,
        "cloud_cover_mean_pct": 50.0,

        # takvim
        "year": int(cal["year"]),
        "month": int(cal["month"]),
        "day": int(cal["day"]),
        "weekday_num": int(cal["weekday_num"]),
        "weekofyear": int(cal["weekofyear"]),
        "quarter": int(cal["quarter"]),
        "is_weekday": int(cal["is_weekday"]),
        "is_weekend": int(cal["is_weekend"]),
        "is_holiday": int(cal["is_holiday"]),
        "is_official_holiday": int(cal["is_official_holiday"]),
        "is_school_day": int(cal["is_school_day"]),
        "Hafta Sonu": int(cal["Hafta Sonu"]),
        "Tatiller": int(cal["Tatiller"]),
        "Okul Günleri": int(cal["Okul Günleri"]),

        # veri setinde varsa diye
        "is_outlier": False,
        "is_extreme_day": 0,

        # opsiyonel bayrak
        "is_religious_holiday": 0,
    }

    return pd.DataFrame([base])


def ensure_required_cols(X: pd.DataFrame, required_cols: list[str]) -> pd.DataFrame:
    if not required_cols:
        return X

    defaults = {
        # sayısal defaultlar
        "rain_mm": 0.0, "precip_mm": 0.0, "snowfall_cm": 0.0, "snow_depth_cm": 0.0, "et0_mm": 0.0,
        "tmax_c": 0.0, "tmin_c": 0.0, "tmean_c": 0.0,
        "tapp_max_c": 0.0, "tapp_min_c": 0.0, "tapp_mean_c": 0.0,
        "wind10m_mean_kmh": 10.0, "cloud_cover_mean_pct": 50.0,
        "sunshine_sec": 0.0, "sunshine_hours": 0.0,
        "passage_cnt": 0.0,
        "year": 0, "month": 0, "day": 0, "weekday_num": 0, "weekofyear": 0, "quarter": 0,
        "Hafta Sonu": 0, "Tatiller": 0, "Okul Günleri": 0,
        "is_weekday": 0, "is_weekend": 0, "is_holiday": 0, "is_school_day": 0,
        "is_official_holiday": 0, "is_religious_holiday": 0,
        "is_extreme_day": 0,

        # kategorik defaultlar
        "station_name": "UNKNOWN",
        "district_name": "UNKNOWN",
        "district_norm": "unknown",
        "date": "1970-01-01",

        # boolean default
        "is_outlier": False,
    }

    for c in required_cols:
        if c not in X.columns:
            X[c] = defaults.get(c, 0)

    # sadece gerekli kolonları sırayla ver (bazı pipeline'lar sıraya duyarlı olabiliyor)
    return X[required_cols]


# =========================
# 6) EKRAN / TAHMİN
# =========================
colA, colB = st.columns([1, 1])

with colA:
    st.subheader("📌 Otomatik Çıkan Bilgiler")
    st.write("**İstasyon:**", station_name)
    st.write("**İlçe:**", district_name)
    st.write("**district_norm:**", district_norm)
    st.write("**Hafta sonu:**", bool(cal["is_weekend"]))
    st.write("**Resmî tatil:**", bool(cal["is_official_holiday"]))
    st.write("**Okul günü:**", bool(cal["is_school_day"]))
    if cal["is_official_holiday"]:
        st.write("**Tatil adı:**", TR_HOLIDAYS.get(d))

with colB:
    st.subheader("🧾 Kullanıcı Girdileri")
    st.write("**Tarih:**", d.strftime("%Y-%m-%d"))
    st.write("**sunshine_hours:**", sunshine_hours)
    st.write("**rain_mm:**", rain_mm)
    st.write("**tmax_c:**", tmax_c)
    st.write("**tmin_c:**", tmin_c)
    st.write("**passage_cnt:**", passage_cnt)

X = build_X()

# Modelin beklediği kolonları bulabiliyorsak ona göre eksikleri tamamla
req_rf = infer_required_columns(rf_pipe)
req_cat = infer_required_columns(cat_pipe)
req_union = list(dict.fromkeys((req_rf or []) + (req_cat or [])))  # union (sıralı)

X_model = ensure_required_cols(X.copy(), req_union) if req_union else X

with st.expander("🔎 Modele giden X (debug)", expanded=False):
    st.dataframe(X_model, use_container_width=True)

if st.button("🚀 Tahmin Et", use_container_width=True):
    try:
        y_rf = np.asarray(rf_pipe.predict(X_model)).reshape(-1)
        y_cat = np.asarray(cat_pipe.predict(X_model)).reshape(-1)
        y = alpha * y_rf + (1 - alpha) * y_cat

        st.success(f"✅ Tahmin (target_day): **{float(y[0]):.4f}**")

        with st.expander("📌 Detay (RF / CatBoost katkısı)", expanded=False):
            st.write("RF:", float(y_rf[0]))
            st.write("CatBoost:", float(y_cat[0]))
            st.write("Alpha:", float(alpha))

    except Exception as e:
        st.error("❌ Tahmin sırasında hata oluştu.")
        st.exception(e)
