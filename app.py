import streamlit as st
import joblib
import pandas as pd
import numpy as np
from datetime import date as dt_date

# CatBoost objesi joblib içinde varsa import şart
# (kullanmasan bile unpickle sırasında lazım olabilir)
from catboost import CatBoostRegressor, CatBoostClassifier  # noqa: F401


# ============================================================
# 1) JOBLIB İÇİN: Custom Class Tanımı (ŞART)
# ============================================================
class RF_CatBoost_Ensemble:
    """
    Joblib ile kaydedilen custom ensemble'ı Streamlit'te açabilmek için
    aynı isimde class tanımı gerekli.

    Bu class, unpickle sonrası __dict__ içine gelen alanları kullanarak
    tahmin üretir (rf + catboost + alpha + opsiyonel preprocessor).
    """

    def _get(self, *names, default=None):
        for n in names:
            if hasattr(self, n):
                return getattr(self, n)
        return default

    def _transform_if_needed(self, X):
        pre = self._get("preprocessor", "preprocess", "prep", default=None)
        if pre is None:
            return X

        X_tr = pre.transform(X)
        if hasattr(X_tr, "toarray"):
            X_tr = X_tr.toarray()
        return X_tr

    def predict(self, X):
        rf = self._get("rf_model", "rf", "random_forest", "model_rf")
        cat = self._get("cat_model", "catboost_model", "cat", "model_cat")
        alpha = self._get("alpha", "weight", "w", default=0.5)

        if rf is None or cat is None:
            raise ValueError(
                "Ensemble içinde rf/catboost modelleri bulunamadı. "
                "Beklenen alanlar: rf_model / cat_model (veya benzerleri)."
            )

        X_in = self._transform_if_needed(X)

        rf_pred = rf.predict(X_in)
        cat_pred = cat.predict(X_in)

        # Şekil uyumluluğu (bazı modeller (n,1) döndürebilir)
        rf_pred = np.asarray(rf_pred).reshape(-1)
        cat_pred = np.asarray(cat_pred).reshape(-1)

        # Yaygın kullanım: alpha*RF + (1-alpha)*CatBoost
        y = alpha * rf_pred + (1 - alpha) * cat_pred
        return y


# ============================================================
# 2) STREAMLIT AYAR
# ============================================================
st.set_page_config(page_title="RF + CatBoost Ensemble Tahmin", layout="wide")
st.title("🚇 RF + CatBoost Ensemble Tahmin")
st.caption("Model: ensemble_rf_catboost.joblib | Hedef: target_day (inputta yok)")

# ============================================================
# 3) MODEL YÜKLE
# ============================================================
@st.cache_resource
def load_model():
    return joblib.load("model/ensemble_rf_catboost.joblib")

model = load_model()

# ============================================================
# 4) KATEGORİK LİSTELER (istersen boş bırak; boşsa text input açar)
# ============================================================
STATIONS = [
    # "Yenikapı", "Aksaray", ...
]
DISTRICTS = [
    # "Fatih", "Kadıköy", ...
]
DISTRICT_NORMS = [
    # "fatih", "kadikoy", ...
]

def select_or_text(label: str, options: list[str]) -> str:
    if options:
        return st.selectbox(label, options)
    return st.text_input(label, value="")

# ============================================================
# 5) SIDEBAR: KATEGORİK + TARİH
# ============================================================
with st.sidebar:
    st.header("🧩 Temel Bilgiler")
    station_name = select_or_text("station_name", STATIONS)

    d = st.date_input("date", value=dt_date(2024, 12, 1))
    date_str = d.strftime("%Y-%m-%d")  # sende object görünüyor → string güvenli

    district_name = select_or_text("district_name", DISTRICTS)
    district_norm = select_or_text("district_norm", DISTRICT_NORMS)

# ============================================================
# 6) FORM
# ============================================================
c1, c2, c3 = st.columns(3)

with c1:
    st.subheader("📅 Takvim Bayrakları (0/1)")
    hafta_sonu = int(st.checkbox("Hafta Sonu", value=False))
    tatiller = int(st.checkbox("Tatiller", value=False))
    okul_gunleri = int(st.checkbox("Okul Günleri", value=False))

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
    is_outlier = st.checkbox("is_outlier", value=False)  # bool

# ============================================================
# 7) MODELE GİDECEK DF (target_day YOK)
# ============================================================
X = pd.DataFrame([{
    "station_name": station_name,
    "date": date_str,
    "Hafta Sonu": hafta_sonu,
    "Tatiller": tatiller,
    "Okul Günleri": okul_gunleri,
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

colA, colB = st.columns([1, 2])

with colA:
    if st.button("Tahmin Et", use_container_width=True):
        try:
            y_pred = model.predict(X)
            st.success(f"✅ Tahmin (target_day): {float(np.asarray(y_pred).reshape(-1)[0]):.4f}")
        except Exception as e:
            st.error("❌ Tahmin sırasında hata oluştu.")
            st.exception(e)

with colB:
    st.info(
        "Kategorik alanlarda (station_name/district_name/district_norm) eğitimde görülmeyen değer girersen "
        "OneHotEncoder hata verebilir. En sağlamı: bu listeleri eğitimdeki unique değerlerle doldurmak."
    )
