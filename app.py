"""Streamlit interface for the COCO movie synopsis classifier."""
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent


@st.cache_resource
def load_model():
    model = joblib.load(ROOT / "models" / "best_rf_model_simplified.pkl")
    vectorizer = joblib.load(ROOT / "models" / "tfidf_vectorizer.pkl")
    encoder = joblib.load(ROOT / "models" / "label_encoder.pkl")
    return model, vectorizer, encoder


@st.cache_data
def load_data():
    return pd.read_csv(ROOT / "data" / "movie_data_cleaned.csv")


def predict_genres(synopsis, model, vectorizer, encoder):
    vector = vectorizer.transform([synopsis])
    probabilities = model.predict_proba(vector)[0]
    indices = probabilities.argsort()[-3:][::-1]
    labels = encoder.inverse_transform(model.classes_[indices])
    return list(zip(labels, probabilities[indices]))


def main():
    st.set_page_config(page_title="Movie Genre Estimator", page_icon="🎬")
    st.title("🎬 Movie Genre Estimator")
    page = st.sidebar.radio("Navigation", ["Home", "Data Overview", "Predict Genre"])

    if page == "Home":
        st.write("영화의 영문 줄거리를 입력하면 가능성이 높은 상위 3개 장르를 예측합니다.")
        st.markdown("**TF-IDF → Random Forest → Streamlit**")
        st.caption("COCO 2024-2 · 영화 줄거리 장르 분류 프로젝트")
    elif page == "Data Overview":
        data = load_data()
        st.subheader("Movie Dataset")
        st.write(f"영화 {len(data):,}개 · 장르 {data['genre'].nunique()}개")
        st.dataframe(data.head(10), use_container_width=True)
    else:
        synopsis = st.text_area("영문 영화 줄거리", height=180)
        if st.button("Predict Genre"):
            if not synopsis.strip():
                st.warning("영화 줄거리를 입력해 주세요.")
                return
            model, vectorizer, encoder = load_model()
            results = predict_genres(synopsis.strip(), model, vectorizer, encoder)
            st.subheader("Top 3 Predicted Genres")
            for rank, (genre, probability) in enumerate(results, 1):
                st.write(f"{rank}. **{genre}** — {probability:.1%}")
            st.caption("표시된 값은 모델의 장르별 예측 확률이며, 검증된 정답 확률을 뜻하지 않습니다.")


if __name__ == "__main__":
    main()
