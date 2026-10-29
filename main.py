
import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


# ==================================================
# 1. 페이지 설정
# ==================================================
st.set_page_config(
    page_title="뇌졸중 예측 실습실",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.title("🧠 뇌졸중 예측 실습실")
st.write(
    "뇌졸중 관련 데이터를 살펴보고, "
    "각 열의 의미와 데이터의 특징을 알아보는 실습 공간입니다."
)

BASE_DIR = Path(__file__).resolve().parent
RECORD_DIR = BASE_DIR / "내_기록"
RECORD_FILE = RECORD_DIR / "데이터_설명.json"

DATA_URL = (
    "https://raw.githubusercontent.com/"
    "greatsong/modudata/main/data/stroke.csv"
)

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


# ==================================================
# 2. 기록 파일 관리
# ==================================================
def load_records():
    """저장된 우리말 뜻과 데이터 출처를 불러옵니다."""
    if not RECORD_FILE.exists():
        return {"meanings": {}, "source": ""}

    try:
        with RECORD_FILE.open("r", encoding="utf-8") as file:
            data = json.load(file)

        if not isinstance(data, dict):
            raise ValueError("JSON 데이터 형식이 올바르지 않습니다.")

        meanings = data.get("meanings", {})
        source = data.get("source", "")

        if not isinstance(meanings, dict):
            meanings = {}

        if not isinstance(source, str):
            source = ""

        return {"meanings": meanings, "source": source}

    except (OSError, ValueError, json.JSONDecodeError):
        st.warning(
            "저장된 기록을 읽을 수 없습니다. "
            "JSON 파일의 형식을 확인해 주세요."
        )
        return {"meanings": {}, "source": ""}


def save_records(meanings, source):
    """기록을 JSON 파일로 저장합니다."""
    RECORD_DIR.mkdir(parents=True, exist_ok=True)

    data = {
        "meanings": meanings,
        "source": source,
    }

    temp_file = RECORD_DIR / "데이터_설명.tmp"

    with temp_file.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

    temp_file.replace(RECORD_FILE)


if "records" not in st.session_state:
    st.session_state.records = load_records()


# ==================================================
# 3. 데이터 불러오기
# ==================================================
st.subheader("1. 데이터 소개")

st.markdown(
    """
    이 데이터에는 사람별 기본 정보와 건강 관련 특성,
    그리고 `stroke` 열의 뇌졸중 발생 여부가 담겨 있습니다.

    먼저 데이터의 크기와 각 열의 특징을 살펴보세요.
    """
)

uploaded_file = st.file_uploader(
    "CSV 파일 업로드 (선택 사항)",
    type=["csv"],
    help="기본 데이터 주소에서 읽지 못할 때 직접 업로드할 수 있습니다.",
)

df = None

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file, encoding="utf-8")
    except (UnicodeDecodeError, pd.errors.ParserError, ValueError) as error:
        st.error(f"업로드한 CSV 파일을 읽지 못했습니다: {error}")
        st.stop()
else:
    try:
        df = pd.read_csv(DATA_URL, encoding="utf-8")
    except Exception as error:
        st.warning(
            "온라인 데이터 파일을 불러오지 못했습니다. "
            "CSV 파일을 직접 업로드해 주세요."
        )
        st.caption(f"불러오기 오류: {error}")

if df is None:
    st.stop()

# 필수 열 확인
missing_columns = [
    column for column in EXPECTED_COLUMNS
    if column not in df.columns
]

if missing_columns:
    st.error(
        "필수 열이 누락되었습니다: "
        + ", ".join(missing_columns)
    )
    st.stop()

if df.empty:
    st.error("데이터에 행이 없습니다.")
    st.stop()

# stroke 열을 숫자로 변환해 집계합니다.
stroke_values = pd.to_numeric(df["stroke"], errors="coerce")
stroke_count = int((stroke_values == 1).sum())
total_people = len(df)
total_columns = len(df.columns)
stroke_ratio = stroke_count / total_people * 100


# ==================================================
# 4. 핵심 숫자 카드
# ==================================================
st.subheader("2. 데이터 한눈에 보기")

card1, card2, card3, card4 = st.columns(4)

card1.metric(
    label="👥 전체 사람 수",
    value=f"{total_people:,}명",
)

card2.metric(
    label="📋 열 개수",
    value=f"{total_columns:,}개",
)

card3.metric(
    label="🧠 stroke가 1인 사람 수",
    value=f"{stroke_count:,}명",
)

card4.metric(
    label="📊 stroke가 1인 비율",
    value=f"{stroke_ratio:.2f}%",
)

st.caption(
    "비율 = stroke가 1인 행의 수 ÷ 전체 행의 수 × 100"
)


# ==================================================
# 5. 열 이름과 데이터 특징
# ==================================================
st.subheader("3. 열 이름과 데이터 특징")

st.write(
    "교재를 참고해 각 열의 우리말 뜻을 직접 작성해 보세요. "
    "우리말 뜻 이외의 열은 수정할 수 없습니다."
)

saved_meanings = st.session_state.records["meanings"]

column_info = pd.DataFrame({
    "열 이름": df.columns.tolist(),
    "우리말 뜻": [
        str(saved_meanings.get(column, ""))
        for column in df.columns
    ],
    "값의 종류": [
        int(df[column].nunique(dropna=True))
        for column in df.columns
    ],
    "빈 값 개수": [
        int(df[column].isna().sum())
        for column in df.columns
    ],
})

edited_info = st.data_editor(
    column_info,
    use_container_width=True,
    hide_index=True,
    num_rows="fixed",
    disabled=[
        "열 이름",
        "값의 종류",
        "빈 값 개수",
    ],
    column_config={
        "열 이름": st.column_config.TextColumn(
            "열 이름",
            width="medium",
        ),
        "우리말 뜻": st.column_config.TextColumn(
            "우리말 뜻",
            width="large",
            help="교재를 참고해 직접 입력하세요.",
        ),
        "값의 종류": st.column_config.NumberColumn(
            "값의 종류",
            help="빈 값을 제외한 고유한 값의 개수",
        ),
        "빈 값 개수": st.column_config.NumberColumn(
            "빈 값 개수",
            help="결측치의 개수",
        ),
    },
    key="column_info_editor",
)


# ==================================================
# 6. 데이터 처음 다섯 줄
# ==================================================
st.subheader("4. 데이터 처음 다섯 줄")

st.dataframe(
    df.head(5),
    use_container_width=True,
    hide_index=True,
)

st.caption(
    "원본 데이터의 처음 다섯 행입니다. "
    "빈 값도 임의로 채우지 않고 원래 상태대로 표시합니다."
)


# ==================================================
# 7. 데이터 출처와 기록 저장
# ==================================================
st.divider()
st.subheader("5. 데이터 출처")

st.write(
    "교재에 안내된 데이터 출처를 아래에 입력하세요. "
    "우리말 뜻과 함께 저장됩니다."
)

source_text = st.text_area(
    "데이터 출처 입력",
    value=st.session_state.records["source"],
    placeholder="교재에 적힌 데이터 출처를 입력하세요.",
    height=120,
    key="source_editor",
)

if st.button(
    "💾 우리말 뜻과 출처 저장",
    type="primary",
    use_container_width=True,
):
    meanings_to_save = {
        str(row["열 이름"]): str(row["우리말 뜻"])
        for _, row in edited_info.iterrows()
    }

    try:
        save_records(meanings_to_save, source_text)

        st.session_state.records = {
            "meanings": meanings_to_save,
            "source": source_text,
        }

        st.success(
            "저장했습니다! 새로고침 후에도 기록을 불러올 수 있습니다."
        )
        st.caption(f"저장 위치: {RECORD_FILE}")

    except OSError as error:
        st.error(f"기록을 저장하지 못했습니다: {error}")


# ==================================================
# 8. 화면 맨 아래 데이터 출처 표시
# ==================================================
st.divider()
st.caption("데이터 출처 기록")

saved_source = st.session_state.records["source"].strip()

if saved_source:
    st.write(saved_source)
else:
    st.info("아직 데이터 출처가 입력되지 않았습니다.")
