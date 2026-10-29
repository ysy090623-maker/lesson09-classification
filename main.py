
import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


# --------------------------------------------------
# 1. 기본 설정
# --------------------------------------------------
st.set_page_config(
    page_title="뇌졸중 예측 실습실",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.title("🧠 뇌졸중 예측 실습실")
st.caption("데이터를 이해하고, 살펴보고, 예측을 준비하는 실습 공간")

BASE_DIR = Path(__file__).resolve().parent
RECORD_DIR = BASE_DIR / "내_기록"
RECORD_FILE = RECORD_DIR / "데이터_설명.json"

EXPECTED_COLUMNS = [
    "id",
    "gender",
    "age",
    "hypertension",
    "heart_disease",
    "ever_married",
    "work_type",
    "Residence_type",
    "avg_glucose_level",
    "bmi",
    "smoking_status",
    "stroke",
]


# --------------------------------------------------
# 2. 기록 불러오기 및 저장하기
# --------------------------------------------------
def load_records():
    """기존 JSON 기록을 불러옵니다."""
    if not RECORD_FILE.exists():
        return {"meanings": {}, "source": ""}

    try:
        with RECORD_FILE.open("r", encoding="utf-8") as file:
            data = json.load(file)

        if not isinstance(data, dict):
            raise ValueError("기록 형식이 올바르지 않습니다.")

        meanings = data.get("meanings", {})
        source = data.get("source", "")

        return {
            "meanings": meanings if isinstance(meanings, dict) else {},
            "source": source if isinstance(source, str) else "",
        }

    except (OSError, json.JSONDecodeError, ValueError):
        st.warning(
            "기록 파일을 읽지 못했습니다. "
            "기존 JSON 파일이 올바른 형식인지 확인해 주세요."
        )
        return {"meanings": {}, "source": ""}


def save_records(meanings, source):
    """우리말 뜻과 데이터 출처를 UTF-8 JSON으로 저장합니다."""
    RECORD_DIR.mkdir(parents=True, exist_ok=True)

    data = {
        "meanings": meanings,
        "source": source,
    }

    with RECORD_FILE.open("w", encoding="utf-8") as file:
        json.dump(
            data,
            file,
            ensure_ascii=False,
            indent=4,
        )


if "saved_records" not in st.session_state:
    st.session_state.saved_records = load_records()


# --------------------------------------------------
# 3. 데이터 불러오기
# --------------------------------------------------
st.subheader("1. 실습 데이터 소개")

st.write(
    "뇌졸중 관련 정보를 담은 표 형식의 데이터입니다. "
    "먼저 데이터의 크기와 각 열의 특징을 확인하고, "
    "교재를 참고해 열의 의미를 직접 정리해 봅니다."
)

uploaded_file = st.file_uploader(
    "CSV 데이터 파일 선택 (UTF-8)",
    type=["csv"],
    help="CSV 파일을 업로드하거나 아래의 기본 경로에 파일을 넣으세요.",
)

default_file = BASE_DIR / "healthcare-dataset-stroke-data.csv"

# 업로드 파일이 있으면 업로드 파일을 우선 사용합니다.
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file, encoding="utf-8")
    except (UnicodeDecodeError, pd.errors.ParserError, ValueError) as error:
        st.error(f"CSV 파일을 읽을 수 없습니다: {error}")
        st.stop()

elif default_file.exists():
    try:
        df = pd.read_csv(default_file, encoding="utf-8")
    except (UnicodeDecodeError, pd.errors.ParserError, ValueError) as error:
        st.error(f"CSV 파일을 읽을 수 없습니다: {error}")
        st.stop()

else:
    st.info(
        "아직 데이터 파일이 없습니다. "
        "위에서 UTF-8 CSV 파일을 업로드하거나 "
        "healthcare-dataset-stroke-data.csv 파일을 "
        "main.py와 같은 폴더에 넣어 주세요."
    )
    st.stop()

# 필수 열 확인
missing_columns = [col for col in EXPECTED_COLUMNS if col not in df.columns]

if missing_columns:
    st.error(
        "데이터에 필요한 열이 없습니다: "
        + ", ".join(missing_columns)
    )
    st.stop()

if df.empty:
    st.warning("데이터에 행이 없습니다.")
    st.stop()

# stroke 값 확인
stroke_values = pd.to_numeric(df["stroke"], errors="coerce")
stroke_count = int((stroke_values == 1).sum())
stroke_ratio = stroke_count / len(df) * 100


# --------------------------------------------------
# 4. 핵심 숫자 카드 4개
# --------------------------------------------------
st.markdown("### 데이터 한눈에 보기")

card1, card2, card3, card4 = st.columns(4)

card1.metric(
    label="👥 전체 사람 수",
    value=f"{len(df):,}명",
)

card2.metric(
    label="📋 전체 열 개수",
    value=f"{len(df.columns):,}개",
)

card3.metric(
    label="🩺 stroke = 1인 사람 수",
    value=f"{stroke_count:,}명",
)

card4.metric(
    label="📊 stroke = 1 비율",
    value=f"{stroke_ratio:.2f}%",
)

st.caption(
    "비율은 전체 행을 기준으로 계산합니다. "
    "stroke 값이 1인 행을 해당 범주로 집계합니다."
)


# --------------------------------------------------
# 5. 열 이름과 특징 표
# --------------------------------------------------
st.markdown("### 열 이름과 데이터 특징")

st.write(
    "우리말 뜻 칸을 클릭해 교재의 설명을 입력하세요. "
    "입력한 내용은 아래의 저장 버튼을 누르면 보관됩니다."
)

saved_meanings = st.session_state.saved_records["meanings"]

column_info = pd.DataFrame({
    "열 이름": df.columns,
    "우리말 뜻": [
        str(saved_meanings.get(col, ""))
        for col in df.columns
    ],
    "값의 종류": [
        int(df[col].nunique(dropna=True))
        for col in df.columns
    ],
    "빈 값 개수": [
        int(df[col].isna().sum())
        for col in df.columns
    ],
})

edited_info = st.data_editor(
    column_info,
    use_container_width=True,
    hide_index=True,
    num_rows="fixed",
    disabled=["열 이름", "값의 종류", "빈 값 개수"],
    column_config={
        "열 이름": st.column_config.TextColumn(
            "열 이름",
            help="원본 데이터의 열 이름",
        ),
        "우리말 뜻": st.column_config.TextColumn(
            "우리말 뜻",
            help="교재를 참고해 각 열의 뜻을 입력하세요.",
            width="large",
        ),
        "값의 종류": st.column_config.NumberColumn(
            "값의 종류",
            help="빈 값을 제외한 서로 다른 값의 개수",
        ),
        "빈 값 개수": st.column_config.NumberColumn(
            "빈 값 개수",
            help="결측치의 개수",
        ),
    },
    key="column_info_editor",
)

# --------------------------------------------------
# 6. 데이터 설명과 출처 저장
# --------------------------------------------------
st.markdown("### 데이터 설명 및 출처 기록")

source_text = st.text_area(
    "데이터 출처",
    value=st.session_state.saved_records["source"],
    placeholder="교재에 적힌 데이터 출처를 여기에 입력하세요.",
    height=100,
    key="source_editor",
)

if st.button("💾 우리말 뜻과 출처 저장", type="primary"):
    meanings_to_save = {
        str(row["열 이름"]): str(row["우리말 뜻"])
        for _, row in edited_info.iterrows()
    }

    try:
        save_records(meanings_to_save, source_text)

        st.session_state.saved_records = {
            "meanings": meanings_to_save,
            "source": source_text,
        }

        st.success(
            f"기록을 저장했습니다: "
            f"{RECORD_FILE.relative_to(BASE_DIR)}"
        )

    except OSError as error:
        st.error(f"파일을 저장하지 못했습니다: {error}")


# --------------------------------------------------
# 7. 데이터 처음 다섯 줄
# --------------------------------------------------
st.markdown("### 데이터 처음 다섯 줄")

st.dataframe(
    df.head(5),
    use_container_width=True,
    hide_index=True,
)

st.caption(
    "위 표는 데이터의 원래 열과 값을 바꾸지 않고 "
    "처음 다섯 행만 표시합니다."
)


# --------------------------------------------------
# 8. 데이터 출처
# --------------------------------------------------
st.divider()
st.markdown("### 데이터 출처")

current_source = st.session_state.saved_records["source"].strip()

if current_source:
    st.write(current_source)
else:
    st.info(
        "아직 출처가 입력되지 않았습니다. "
        "위의 데이터 출처 칸에 교재의 내용을 입력하고 저장해 주세요."
    )

st.caption("우리말 뜻과 데이터 출처는 내_기록/데이터_설명.json에 저장됩니다.")
