import os
import streamlit as st
import joblib
import pandas as pd
import numpy as np
from datetime import date as dt_date

from catboost import CatBoostRegressor, CatBoostClassifier  # noqa: F401


# =========================
# SABİT AĞIRLIK
# =========================
ALPHA = 0.7  # 0.7 RF + 0.3 CatBoost


# ============================================================
# Custom class (unpickle için gerekebilir)
# ============================================================
class RF_CatBoost_Ensemble:
    """
    Kendi oluşturduğun RF+CatBoost ensemble joblib ile kaydedildiyse,
    aynı class adı burada olmalı.

    Bu class:
    - RF ve CatBoost'u attribute isminden bağımsız TYPE ile bulur
    - preprocessor varsa uygular
    - sabit ALPHA ile birleştirir: ALPHA*RF + (1-ALPHA)*CAT
    """

    def _items(self):
        return list(getattr(self, "__dict__", {}).items())

    def _is_catboost(self, obj) -> bool:
        if obj is None:
            return False
        cls = obj.__class__.__name__.lower()
        mod = getattr(obj.__class__, "__module__", "").lower()
        return ("catboost" in cls) or ("catboost" in mod)

    def _is_rf_like(self, obj) -> bool:
        if obj is None:
            return False
        cls = obj.__class__.__name__.lower()
        mod = getattr(obj.__class__, "__module__", "").lower()
        if "randomforest" in cls or "extratrees" in cls:
            return True
        if "sklearn" in mod and ("ensemble" in mod or "forest" in mod):
            return hasattr(obj, "predict")
        return False

    def _pick_preprocessor(self):
        # En sık kullanılan isimler
        for k in ["preprocessor", "preprocess", "prep", "column_transformer", "ct"]:
            if hasattr(self, k):
                return getattr(self, k)
        return None

    def _transform(self, X):
        pre = self._pick_preprocessor()
        if pre is None:
            return X
        Xt = pre.transform(X)
        if hasattr(Xt, "toarray"):
            Xt = Xt.toarray()
        return Xt

    def _find_models(self):
        rf = None
        cat = None

        # 1) Direkt alanlarda ara
        for name, obj in self._items():
            if cat is None and self._is_catboost(obj):
                cat = obj
            if rf is None and self._is_rf_like(obj):
                rf = obj
            if rf is not None and cat is not None:
                return rf, cat

        # 2) List/dict içlerine de bak
        for name, obj in self._items():
            if isinstance(obj, dict):
                for _, v in obj.items():
                    if cat is None and self._is_catboost(v):
                        cat = v
                    if rf is None and self._is_rf_like(v):
                        rf = v
                    if rf is not None and cat is not None:
                        return rf, cat
            if isinstance(obj, (list, tuple)):
                for v in obj:
                    if cat is None and self._is_catboost(v):
                        cat = v
                    if rf is None and self._is_rf_like(v):
                        rf = v
                    if rf is not None and cat is not None:
                        return rf, cat

        return rf, cat

    def predict(self, X):
        rf, cat = self._find_models()
        if rf is None or cat is None:
            keys = [k for k, _ in self._items()]
            raise ValueError(
                "Ensemble içinde RF/CatBoost bulunamadı. "
                f"Mevcut anahtarlar: {keys}"
            )

        Xt = self._transform(X)
        rf_pred = np.asarray(rf.predict(Xt)).reshape(-1)
        cat_pred = np.asarray(cat.predict(Xt)).reshape(-1)

        return ALPHA * rf_pred + (1 - ALPHA) * cat_pred


# ============================================================
# Streamlit
# ============================================================
st.set_page_config(page_title="RF(0.7) + CatBoost(0.3) Tahmin", layout="wide")
st.title("🚇 RF(0.7) + CatBoost(0.3) Ensemble Tahmin")
st.caption("Hedef: target_day (inputta yok)")

MODEL_PATH = "ensemble_rf_catboost.joblib"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

# Debug
with st.expander("🔧 Debug", expanded=False):
    st.write("Working dir:", os.getcwd())
    st.write("Files:", os.listdir("."))
    st.write("Model exists:", os.path.exists(MODEL_PATH))
    if os.path.exists(MODEL_PATH):
        st.write("Model size (MB):", round(os.path.getsize(MODEL_PATH)/(1024*1024), 2))

model = load_model()

with st.expander("🔍 Debug (model anahtarları)", expanded=False):
    st.write("Model type:", type(model))
    if hasattr(model, "__dict__"):
        st.write("Top-level keys:", list(model.__dict__.keys()))


# ============================================================
# INPUTS (senin kolonlara göre)
# ============================================================
STATIONS = []
DISTRICTS = []
DISTRICT_NORMS = []

def select_or_text(label: str, options: list[str]) -> str:
    return st.selectbox(label, options) if options else st.text_input(label, value="")

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

if st.button("Tahmin Et", use_container_width=True):
    try:
        y_pred = model.predict(X)
        y0 = float(np.asarray(y_pred).reshape(-1)[0])
        st.success(f"✅ Tahmin (target_day): {y0:.4f}")
        st.caption(f"Ensemble ağırlığı: {ALPHA:.1f} RF + {1-ALPHA:.1f} CatBoost")
    except Exception as e:
        st.error("❌ Tahmin sırasında hata oluştu.")
        st.exception(e)
