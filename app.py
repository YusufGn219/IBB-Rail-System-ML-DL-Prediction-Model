import os
import streamlit as st
import joblib
import pandas as pd
import numpy as np
from datetime import date as dt_date

# joblib içinde catboost objesi varsa import şart olabilir
from catboost import CatBoostRegressor, CatBoostClassifier  # noqa: F401


# ============================================================
# CUSTOM ENSEMBLE CLASS (unpickle için gerekebilir)
# ============================================================
class RF_CatBoost_Ensemble:
    """
    Joblib ile kaydedilen custom ensemble'ı Streamlit'te açabilmek için
    aynı class ismi gerekli.

    Bu sürüm attribute isimlerine bağlı kalmaz; objeleri TYPE ile bulur.
    """

    # ---- yardımcılar ----
    def _all_items(self):
        """(name, obj) listesi döndürür; sadece __dict__ içinden."""
        try:
            return list(self.__dict__.items())
        except Exception:
            return []

    def _is_catboost(self, obj) -> bool:
        if obj is None:
            return False
        cls = obj.__class__.__name__.lower()
        mod = getattr(obj.__class__, "__module__", "").lower()
        return ("catboost" in cls) or ("catboost" in mod)

    def _is_random_forest_like(self, obj) -> bool:
        if obj is None:
            return False
        cls = obj.__class__.__name__.lower()
        mod = getattr(obj.__class__, "__module__", "").lower()
        # RandomForest / ExtraTrees gibi forest modelleri
        if "randomforest" in cls or "extratrees" in cls:
            return True
        # sklearn ensemble/forest modülleri
        if ("sklearn" in mod) and (("ensemble" in mod) or ("forest" in mod)):
            # ağaç/forest tahminleyici olsun
            return hasattr(obj, "predict")
        return False

    def _pick_alpha(self) -> float:
        # alpha farklı isimlerle kaydedilmiş olabilir
        candidates = ["alpha", "best_alpha", "weight", "w", "rf_weight"]
        for k in candidates:
            if hasattr(self, k):
                try:
                    a = float(getattr(self, k))
                    if 0.0 <= a <= 1.0:
                        return a
                except Exception:
                    pass
        return 0.5

    def _pick_preprocessor(self):
        # preprocess de farklı isimlerle olabilir
        candidates = ["preprocessor", "preprocess", "prep", "column_transformer", "ct"]
        for k in candidates:
            if hasattr(self, k):
                return getattr(self, k)
        return None

    def _transform_if_needed(self, X):
        pre = self._pick_preprocessor()
        if pre is None:
            return X
        X_tr = pre.transform(X)
        if hasattr(X_tr, "toarray"):
            X_tr = X_tr.toarray()
        return X_tr

    def _find_models(self):
        """
        İçerikten rf benzeri ve catboost modeli bulmaya çalışır.
        1) Direkt objeler
        2) Liste/dict içinde saklı objeler
        """
        rf = None
        cat = None

        # 1) __dict__ içindeki doğrudan objeler
        for name, obj in self._all_items():
            if cat is None and self._is_catboost(obj):
                cat = obj
            if rf is None and self._is_random_forest_like(obj):
                rf = obj
            if rf is not None and cat is not None:
                return rf, cat

        # 2) İçeride dict/list/tuple varsa onların içine de bak
        for name, obj in self._all_items():
            if isinstance(obj, dict):
                for k2, v2 in obj.items():
                    if cat is None and self._is_catboost(v2):
                        cat = v2
                    if rf is None and self._is_random_forest_like(v2):
                        rf = v2
                    if rf is not None and cat is not None:
                        return rf, cat
            elif isinstance(obj, (list, tuple)):
                for v2 in obj:
                    if cat is None and self._is_catboost(v2):
                        cat = v2
                    if rf is None and self._is_random_forest_like(v2):
                        rf = v2
                    if rf is not None and cat is not None:
                        return rf, cat

        return rf, cat

    # ---- ana predict ----
    def predict(self, X):
        rf, cat = self._find_models()
        if rf is None or cat is None:
            # Debug için: hangi anahtarlar var?
            keys = [k for k, _ in self._all_items()]
            raise ValueError(
                "Ensemble içinde RF/CatBoost modelleri otomatik bulunamadı.\n"
                f"Mevcut anahtarlar: {keys}\n"
                "Çözüm: Debug ekranında görünen anahtarlara göre arama kurallarını genişletiriz."
            )

        alpha = self._pick_alpha()

        # preprocess varsa uygula
        X_in = self._transform_if_needed(X)

        rf_pred = np.asarray(rf.predict(X_in)).reshape(-1)
        cat_pred = np.asarray(cat.predict(X_in)).reshape(-1)

        return alpha * rf_pred + (1 - alpha) * cat_pred


# ============================================================
# STREAMLIT
# ============================================================
st.set_page_config(page_title="RF + CatBoost Ensemble Tahmin", layout="wide")
st.title("🚇 RF + CatBoost Ensemble Tahmin")

MODEL_PATH = "ensemble_rf_catboost.joblib"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

# Dosya kontrol
with st.expander("🔧 Debug (dosya + model içeriği)", expanded=False):
    st.write("Working dir:", os.getcwd())
    st.write("Files:", os.listdir("."))
    st.write("Model exists:", os.path.exists(MODEL_PATH))
    if os.path.exists(MODEL_PATH):
        st.write("Model size (MB):", round(os.path.getsize(MODEL_PATH) / (1024 * 1024), 2))

model = load_model()

with st.expander("🔍 Debug (ensemble anahtarları)", expanded=False):
    st.write("Loaded model type:", type(model))
    if hasattr(model, "__dict__"):
        st.write("Top-level keys:", list(model.__dict__.keys()))
        # Muhtemel rf/cat objelerini göster
        hits = []
        for k, v in model.__dict__.items():
            if v is None:
                continue
            cls = v.__class__.__name__
            mod = getattr(v.__class__, "__module__", "")
            if ("catboost" in cls.lower()) or ("catboost" in mod.lower()) or ("randomforest" in cls.lower()) or ("extratrees" in cls.lower()):
                hits.append((k, cls, mod))
        if hits:
            st.write("Şüpheli model objeleri (key, class, module):")
            st.write(hits)
    else:
        st.write("Model __dict__ yok (beklenmeyen durum).")

st.caption("Hedef: target_day (inputta yok)")

# ============================================================
# INPUTLAR (senin kolon listene göre)
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
    except Exception as e:
        st.error("❌ Tahmin sırasında hata oluştu.")
        st.exception(e)
