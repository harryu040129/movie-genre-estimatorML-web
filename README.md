# movie-genre-estimatorML-web

**영화 줄거리를 분석해 장르를 예측하는 Streamlit 웹 애플리케이션입니다.**

COCO 2024학년도 2학기 프로젝트로, 영화 텍스트를 TF-IDF로 벡터화하고 Random Forest로 분류한 뒤 웹 화면에서 상위 3개 예측 장르를 확인하도록 구현했습니다.

## 주요 기능

- **Home** — 프로젝트 소개
- **Data Overview** — 정제된 영화 데이터와 샘플 확인
- **Predict Genre** — 영문 줄거리를 입력해 상위 3개 장르와 모델 예측 확률 확인
- 데이터와 모델 캐싱으로 화면 변경 시 불필요한 재로딩 방지

## 예측 흐름

```text
영화 줄거리 → TF-IDF 벡터화 → Random Forest → 장르별 확률 → Top 3 결과
```

## 프로젝트 구조

```text
movie-genre-estimatorML-web/
├── app.py                       # Streamlit 웹 앱
├── train_model.ipynb             # 모델 학습·평가 기록
├── data/
│   └── movie_data_cleaned.csv    # 정제된 영화 데이터
├── models/
│   ├── best_rf_model_simplified.pkl
│   ├── tfidf_vectorizer.pkl
│   └── label_encoder.pkl
├── requirements.txt
└── README.md
```

## 실행 방법

Python 3.10 또는 3.11 환경을 권장합니다. 저장 모델에는 scikit-learn 1.4.2 버전 기록이 있어 해당 버전을 사용합니다.

```bash
git clone https://github.com/harryu040129/movie-genre-estimatorML-web.git
cd movie-genre-estimatorML-web
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
streamlit run app.py
```

Windows에서는 가상환경을 `.venv\Scripts\activate`로 활성화합니다. 모델과 데이터 경로는 `app.py` 위치를 기준으로 처리합니다.

## 데이터와 모델

| 항목 | 내용 |
|---|---|
| 데이터 | 영화 53,990행, 제목·줄거리·장르 등 4개 컬럼 |
| 모델 입력 | 영문 영화 줄거리 (`synopsis`) |
| 분류 대상 | action, adventure, crime, family, fantasy, horror, mystery, romance, scifi, thriller |
| 벡터화 | TF-IDF, 최대 5,000개 특성, 영문 불용어 제거 |
| 학습 노트북 설정 | Random Forest, 트리 100개, 최대 깊이 20 |
| 학습/테스트 분할 | 80% / 20%, random_state=42 |
| 저장된 노트북 평가 결과 | Accuracy 0.29 |

정확도 0.29는 원본 노트북에 저장된 단일 장르 예측 결과이며, 이번 정리 과정에서 재측정한 값이나 Top-3 정확도가 아닙니다. 저장된 모델의 정확한 학습 실행 이력은 별도로 검증하지 않았습니다.

## 학습 노트북

저장소 루트에서 다음 명령으로 열 수 있습니다.

```bash
python -m pip install jupyterlab
jupyter lab train_model.ipynb
```

노트북을 다시 실행하면 `models/`의 모델 파일을 새 학습 결과로 덮어씁니다. 기존 출력은 과거 수업 실행 기록입니다.

## 기술

Python · Streamlit · Pandas · scikit-learn · TF-IDF · Random Forest · joblib

## 참고 사항

- 수업 자료, 개인 연습 코드, 중복 모델 버전, 압축본은 제외하고 핵심 프로젝트만 구성했습니다.
- 원본 학습 코드는 전체 텍스트에 TF-IDF를 적용한 뒤 데이터를 나눕니다. 엄격한 성능 평가에는 학습 데이터로만 벡터라이저를 fit하는 개선이 필요합니다.
- 데이터 최초 배포 출처와 라이선스는 원본 자료에서 확인되지 않아 임의의 오픈소스 라이선스를 지정하지 않았습니다.
