import os
import streamlit as st
import joblib
import pandas as pd
import numpy as np
from datetime import date as dt_date

import holidays  # pip: holidays

# =========================
# SAYFA AYARLARI
# =========================
st.set_page_config(page_title="RF + CatBoost Ensemble Tahmin", layout="wide")
st.title("🚇 RF + CatBoost Ensemble Tahmin (Bundle)")

# =========================
# BUNDLE YÜKLEME
# =========================
BUNDLE_PATH = "bundle_rf_catboost.joblib"

@st.cache_resource
def load_bundle(path: str):
    return joblib.load(path)

with st.expander("🔧 Debug", expanded=False):
    st.write("Working dir:", os.getcwd())
    st.write("Files:", os.listdir("."))
    st.write("Bundle exists:", os.path.exists(BUNDLE_PATH))
    if os.path.exists(BUNDLE_PATH):
        st.write("Bundle size (MB):", round(os.path.getsize(BUNDLE_PATH)/(1024*1024), 2))

if not os.path.exists(BUNDLE_PATH):
    st.error(
        f"❌ `{BUNDLE_PATH}` bulunamadı.\n\n"
        "Repo kök dizinine `bundle_rf_catboost.joblib` dosyasını koy ve tekrar deploy et."
    )
    st.stop()

bundle = load_bundle(BUNDLE_PATH)

alpha = float(bundle.get("alpha", 0.7))
rf_pipe = bundle["rf_pipe"]
cat_pipe = bundle["cat_pipe"]

st.caption(f"Ağırlıklar: **{alpha:.2f} RF** + **{1-alpha:.2f} CatBoost**")


# =========================
# TAKVİM / TATİL HESAPLARI
# =========================
@st.cache_resource
def tr_holidays():
    # Türkiye resmi tatilleri
    return holidays.Turkey()

TR_HOLIDAYS = tr_holidays()

def compute_calendar_features(d: dt_date):
    # temel
    weekday_num = d.weekday()  # Mon=0 ... Sun=6
    is_weekend = int(weekday_num >= 5)
    is_weekday = int(not is_weekend)

    # yıl/ay/gün
    year = d.year
    month = d.month
    day = d.day

    # weekofyear (ISO hafta numarası)
    weekofyear = int(d.isocalendar().week)

    # quarter
    quarter = (month - 1) // 3 + 1

    # TR resmi tatil mi?
    is_official_holiday = int(d in TR_HOLIDAYS)
    is_holiday = int(is_official_holiday == 1)  # senin kolon mantığına göre

    # “Tatiller” kolonunu resmi tatile bağlayalım
    Tatiller = is_official_holiday

    # “Hafta Sonu” kolonunu otomatik dolduralım
    Hafta_Sonu = is_weekend

    return {
        "weekday_num": weekday_num,
        "is_weekend": is_weekend,
        "is_weekday": is_weekday,
        "year": year,
        "month": month,
        "day": day,
        "weekofyear": weekofyear,
        "quarter": quarter,
        "is_official_holiday": is_official_holiday,
        "is_holiday": is_holiday,
        "Tatiller": Tatiller,
        "Hafta Sonu": Hafta_Sonu,
    }


# =========================
# INPUTLAR
# =========================
STATIONS = []
DISTRICTS = []
DISTRICT_NORMS = []

def select_or_text(label: str, options: list[str], default=""):
    if options:
        return st.selectbox(label, options)
    return st.text_input(label, value=default)

with st.sidebar:
    st.header("🧩 Temel Bilgiler")
    station_name = select_or_text("station_name", STATIONS)

    d = st.date_input("date", value=dt_date(2024, 12, 1))
    date_str = d.strftime("%Y-%m-%d")

    district_name = select_or_text("district_name", DISTRICTS)
    district_norm = select_or_text("district_norm", DISTRICT_NORMS)

cal = compute_calendar_features(d)

with st.sidebar:
    st.markdown("### 📅 Otomatik Takvim Bilgileri")
    st.write("Weekday num:", cal["weekday_num"])
    st.write("Week of year:", cal["weekofyear"])
    st.write("Quarter:", cal["quarter"])
    st.write("Hafta sonu mu?:", bool(cal["is_weekend"]))
    st.write("Resmi tatil mi?:", bool(cal["is_official_holiday"]))
    if cal["is_official_holiday"]:
        st.write("Tatil adı:", TR_HOLIDAYS.get(d))

c1, c2, c3 = st.columns(3)

with c1:
    st.subheader("📅 Takvim Bayrakları")
    # otomatik dolduruluyor; kullanıcı isterse override edebilsin diye checkbox koydum
    override = st.checkbox("Takvim bayraklarını manuel değiştireyim", value=False)

    if override:
        Hafta_Sonu = int(st.checkbox("Hafta Sonu", value=bool(cal["Hafta Sonu"])))
        Tatiller = int(st.checkbox("Tatiller", value=bool(cal["Tatiller"])))
        is_weekday = int(st.checkbox("is_weekday", value=bool(cal["is_weekday"])))
        is_weekend = int(st.checkbox("is_weekend", value=bool(cal["is_weekend"])))
        is_holiday = int(st.checkbox("is_holiday", value=bool(cal["is_holiday"])))
        is_official_holiday = int(st.checkbox("is_official_holiday", value=bool(cal["is_official_holiday"])))
    else:
        Hafta_Sonu = cal["Hafta Sonu"]
        Tatiller = cal["Tatiller"]
        is_weekday = cal["is_weekday"]
        is_weekend = cal["is_weekend"]
        is_holiday = cal["is_holiday"]
        is_official_holiday = cal["is_official_holiday"]

    # Okul günleri / dini tatil gibi şeyler otomatik değil -> kullanıcı girsin
    Okul_Gunleri = int(st.checkbox("Okul Günleri", value=1))
    is_school_day = int(st.checkbox("is_school_day", value=1))
    is_religious_holiday = int(st.checkbox("is_religious_holiday", value=0))

with c2:
    st.subheader("🌦️ Hava Durumu")
    rain_mm = st.number_input("rain_mm", value=0.0, step=0.1)
    precip_mm = st.number_input("precip_mm", value=0.0, step=0.1)
    snowfall_cm = st.number_input("snowfall_cm", value=0.0, step=0.1)
    snow_depth_cm = st.number_input("snow_depth_cm", value=0.0, step=0.1)
    et0_mm = st.number_input("et0_mm", value=0.0, step=0.1)

    tmax_c = st.number_input("tmax_c", value=20.0, step=0.1)
    tmin_c = st.number_input("tmin_c", value=10.0, step=0.1)
    tmean_c = st.number_input("tmean_c", value=15.0, step=0.1)

    tapp_max_c = st.number_input("tapp_max_c", value=20.0, step=0.1)
    tapp_min_c = st.number_input("tapp_min_c", value=10.0, step=0.1)
    tapp_mean_c = st.number_input("tapp_mean_c", value=15.0, step=0.1)

    wind10m_mean_kmh = st.number_input("wind10m_mean_kmh", value=10.0, step=0.1)
    cloud_cover_mean_pct = st.number_input("cloud_cover_mean_pct", value=50.0, step=0.1)

    sunshine_sec = st.number_input("sunshine_sec", value=0.0, step=1.0)
    sunshine_hours = st.number_input("sunshine_hours", value=0.0, step=0.1)

with c3:
    st.subheader("🧠 Diğerleri")
    passage_cnt = st.number_input("passage_cnt", value=0.0, step=1.0)

    # otomatik hesaplanan zaman alanları
    year = cal["year"]
    month = cal["month"]
    day = cal["day"]
    weekday_num = cal["weekday_num"]
    weekofyear = cal["weekofyear"]
    quarter = cal["quarter"]

    # diğer bayraklar
    is_extreme_day = int(st.checkbox("is_extreme_day", value=False))
    is_outlier = st.checkbox("is_outlier", value=False)

    st.caption("Not: year/month/day/weekday_num/weekofyear/quarter otomatik hesaplanır.")


# =========================
# X OLUŞTUR
# =========================
X = pd.DataFrame([{
    "station_name": station_name,
    "date": date_str,

    "Hafta Sonu": int(Hafta_Sonu),
    "Tatiller": int(Tatiller),
    "Okul Günleri": int(Okul_Gunleri),

    "passage_cnt": float(passage_cnt),

    "rain_mm": float(rain_mm),
    "precip_mm": float(precip_mm),
    "snowfall_cm": float(snowfall_cm),
    "et0_mm": float(et0_mm),

    "tmax_c": float(tmax_c),
    "tmin_c": float(tmin_c),
    "tmean_c": float(tmean_c),

    "tapp_max_c": float(tapp_max_c),
    "tapp_min_c": float(tapp_min_c),
    "tapp_mean_c": float(tapp_mean_c),

    "wind10m_mean_kmh": float(wind10m_mean_kmh),
    "cloud_cover_mean_pct": float(cloud_cover_mean_pct),

    "sunshine_sec": float(sunshine_sec),
    "sunshine_hours": float(sunshine_hours),

    "snow_depth_cm": float(snow_depth_cm),

    "year": int(year),
    "month": int(month),
    "day": int(day),
    "weekday_num": int(weekday_num),
    "weekofyear": int(weekofyear),
    "quarter": int(quarter),

    "is_weekday": int(is_weekday),
    "is_weekend": int(is_weekend),
    "is_holiday": int(is_holiday),
    "is_school_day": int(is_school_day),

    "is_outlier": bool(is_outlier),
    "is_extreme_day": int(is_extreme_day),

    "is_official_holiday": int(is_official_holiday),
    "is_religious_holiday": int(is_religious_holiday),

    "district_name": district_name,
    "district_norm": district_norm,
}])

st.subheader("🔎 Modele giden veri (kontrol)")
st.dataframe(X, use_container_width=True)

# =========================
# TAHMİN
# =========================
if st.button("Tahmin Et", use_container_width=True):
    try:
        y_rf = np.asarray(rf_pipe.predict(X)).reshape(-1)
        y_cat = np.asarray(cat_pipe.predict(X)).reshape(-1)
        y = alpha * y_rf + (1 - alpha) * y_cat

        st.success(f"✅ Tahmin (target_day): {float(y[0]):.4f}")

        with st.expander("📌 Detay", expanded=False):
            st.write("RF:", float(y_rf[0]))
            st.write("CatBoost:", float(y_cat[0]))
            st.write("Alpha:", alpha)

    except Exception as e:
        st.error("❌ Tahmin sırasında hata oluştu.")
        st.exception(e)
