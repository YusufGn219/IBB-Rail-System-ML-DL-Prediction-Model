import os
import streamlit as st
import joblib
import pandas as pd
import numpy as np
from datetime import date as dt_date


# =========================
# SAYFA AYARLARI
# =========================
st.set_page_config(page_title="RF + CatBoost Ensemble Tahmin", layout="wide")
st.title("🚇 RF + CatBoost Ensemble Tahmin")


# =========================
# BUNDLE YÜKLEME
# =========================
BUNDLE_PATH = "bundle_rf_catboost.joblib"

@st.cache_resource
def load_bundle():
    return joblib.load(BUNDLE_PATH)

with st.expander("🔧 Debug (dosya kontrol)", expanded=False):
    st.write("Working dir:", os.getcwd())
    st.write("Files:", os.listdir("."))
    st.write("Bundle exists:", os.path.exists(BUNDLE_PATH))
    if os.path.exists(BUNDLE_PATH):
        st.write("Bundle size (MB):", round(os.path.getsize(BUNDLE_PATH) / (1024 * 1024), 2))

if not os.path.exists(BUNDLE_PATH):
    st.error(
        f"❌ `{BUNDLE_PATH}` bulunamadı.\n\n"
        "Repo kök dizinine `bundle_rf_catboost.joblib` dosyasını koyduğundan emin ol."
    )
    st.stop()

bundle = load_bundle()

# bundle içeriği
alpha = float(bundle.get("alpha", 0.7))
rf_pipe = bundle["rf_pipe"]
cat_pipe = bundle["cat_pipe"]

st.caption(f"Ensemble ağırlıkları: **{alpha:.2f} RF** + **{1-alpha:.2f} CatBoost**")


# =========================
# PIPELINE KOLONLARI (varsa) - KONTROL
# =========================
def pipeline_expected_columns(pipe):
    """
    Eğer pipeline bir ColumnTransformer veya benzeri bir şey kullanıyorsa,
    beklenen kolonları buradan yakalamaya çalışır.
    Bulamazsa None döner.
    """
    # sklearn pipeline: named_steps olabilir
    steps = getattr(pipe, "named_steps", None)
    if not steps:
        return None

    # en sık: "preprocessor" adında adım olur
    for k in ["preprocessor", "prep", "ct", "column_transformer"]:
        if k in steps:
            pre = steps[k]
            # ColumnTransformer içindeki feature isimleri bazen saklı olur
            for attr in ["feature_names_in_", "feature_names_in"]:
                if hasattr(pre, attr):
                    return list(getattr(pre, attr))
            # pipeline'ın en üstü bazen feature_names_in_ tutar
    for attr in ["feature_names_in_", "feature_names_in"]:
        if hasattr(pipe, attr):
            return list(getattr(pipe, attr))

    return None

exp_cols_rf = pipeline_expected_columns(rf_pipe)
exp_cols_cat = pipeline_expected_columns(cat_pipe)

with st.expander("🧠 Debug (pipeline beklenen kolonlar)", expanded=False):
    st.write("RF pipe type:", type(rf_pipe))
    st.write("Cat pipe type:", type(cat_pipe))
    st.write("RF expected columns:", exp_cols_rf if exp_cols_rf else "Bulunamadı (normal olabilir)")
    st.write("Cat expected columns:", exp_cols_cat if exp_cols_cat else "Bulunamadı (normal olabilir)")


# =========================
# INPUTLAR (senin kolonlarına göre)
# =========================
# (Listeleri boş bırakırsan text_input olur)
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

c1, c2, c3 = st.columns(3)

with c1:
    st.subheader("📅 Takvim Bayrakları (0/1)")
    Hafta_Sonu = int(st.checkbox("Hafta Sonu", value=False))
    Tatiller = int(st.checkbox("Tatiller", value=False))
    Okul_Gunleri = int(st.checkbox("Okul Günleri", value=False))

    is_weekday = int(st.checkbox("is_weekday", value=True))
    is_weekend = int(st.checkbox("is_weekend", value=False))
    is_holiday = int(st.checkbox("is_holiday", value=False))
    is_school_day = int(st.checkbox("is_school_day", value=True))

    is_official_holiday = int(st.checkbox("is_official_holiday", value=False))
    is_religious_holiday = int(st.checkbox("is_religious_holiday", value=False))

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
    sunshine_hour = st.number_input("sunshine_hour", value=0.0, step=0.1)

with c3:
    st.subheader("🧠 Zaman Özellikleri + Diğerleri")
    passage_cnt = st.number_input("passage_cnt", value=0.0, step=1.0)

    year = st.number_input("year", value=d.year, step=1)
    month = st.number_input("month", value=d.month, step=1, min_value=1, max_value=12)
    day = st.number_input("day", value=d.day, step=1, min_value=1, max_value=31)

    weekday_num = st.number_input("weekday_num", value=d.weekday(), step=1, min_value=0, max_value=6)
    weekofyear = st.number_input("weekofyear", value=int(d.strftime("%U")), step=1, min_value=0, max_value=53)
    quarter = st.number_input("quarter", value=((d.month - 1) // 3) + 1, step=1, min_value=1, max_value=4)

    is_extreme_day = int(st.checkbox("is_extreme_day", value=False))
    is_outlier = st.checkbox("is_outlier", value=False)

# modele giden veri
X = pd.DataFrame([{
    "station_name": station_name,
    "date": date_str,

    "Hafta Sonu": Hafta_Sonu,
    "Tatiller": Tatiller,
    "Okul Günleri": Okul_Gunleri,

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
    "sunshine_hour": float(sunshine_hour),
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

st.divider()
st.subheader("🔎 Modele giden veri (kontrol)")
st.dataframe(X, use_container_width=True)


# =========================
# KOLON UYUMLULUK KONTROLÜ
# =========================
def show_column_mismatch(expected_cols, X_cols, title):
    exp_set = set(expected_cols)
    x_set = set(X_cols)
    missing = sorted(list(exp_set - x_set))
    extra = sorted(list(x_set - exp_set))

    if not missing and not extra:
        st.success(f"✅ {title}: Kolonlar uyumlu.")
        return True

    st.warning(f"⚠️ {title}: Kolon uyuşmazlığı var.")
    if missing:
        st.write("Eksik kolonlar:", missing)
    if extra:
        st.write("Fazla kolonlar:", extra)
    return False


# =========================
# TAHMİN
# =========================
if st.button("Tahmin Et", use_container_width=True):
    try:
        # Eğer pipeline expected kolonları bulabildiysek uyumluluğu kontrol et
        if exp_cols_rf:
            ok_rf = show_column_mismatch(exp_cols_rf, list(X.columns), "RF Pipeline")
            if not ok_rf:
                st.stop()

        if exp_cols_cat:
            ok_cat = show_column_mismatch(exp_cols_cat, list(X.columns), "CatBoost Pipeline")
            if not ok_cat:
                st.stop()

        y_rf = np.asarray(rf_pipe.predict(X)).reshape(-1)
        y_cat = np.asarray(cat_pipe.predict(X)).reshape(-1)

        y_pred = alpha * y_rf + (1 - alpha) * y_cat
        y0 = float(y_pred[0])

        st.success(f"✅ Tahmin (target_day): {y0:.4f}")
        st.caption(f"Ensemble: {alpha:.2f}*RF + {1-alpha:.2f}*CatBoost")

        with st.expander("📌 Model çıktı detayları", expanded=False):
            st.write("RF tahmin:", float(y_rf[0]))
            st.write("CatBoost tahmin:", float(y_cat[0]))

    except Exception as e:
        st.error("❌ Tahmin sırasında hata oluştu.")
        st.exception(e)
