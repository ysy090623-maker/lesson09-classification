
import pandas as pd
import plotly.express as px
import streamlit as st


# ==================================================
# 1. 페이지 설정
# ==================================================
st.set_page_config(
    page_title="탐색 | 뇌졸중 예측 실습실",
    page_icon="🔎",
    layout="wide",
)

st.title("🔎 데이터 탐색")
st.write(
    "나이, 평균 혈당, 고혈압, 심장병, 체질량지수, "
    "흡연 상태와 뇌졸중의 관계를 탐색합니다."
)

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
# 2. 데이터 불러오기
# ==================================================
@st.cache_data
def load_data():
    """지정된 주소에서 UTF-8 CSV 데이터를 읽습니다."""
    data = pd.read_csv(DATA_URL, encoding="utf-8")

    missing_columns = [
        column
        for column in EXPECTED_COLUMNS
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            "필수 열이 누락되었습니다: "
            + ", ".join(missing_columns)
        )

    return data


try:
    df = load_data()
except Exception as error:
    st.error(f"데이터를 불러오지 못했습니다: {error}")
    st.info(
        "인터넷 연결과 데이터 주소를 확인한 다음 "
        "페이지를 새로고침해 주세요."
    )
    st.stop()

# 분석에 사용할 숫자형 열을 안전하게 변환합니다.
numeric_columns = [
    "age",
    "avg_glucose_level",
    "bmi",
    "hypertension",
    "heart_disease",
    "stroke",
]

for column in numeric_columns:
    df[column] = pd.to_numeric(df[column], errors="coerce")

# 분석용 뇌졸중 범주를 만듭니다.
df["stroke_group"] = df["stroke"].map({
    0: "뇌졸중 없음",
    1: "뇌졸중 있음",
})

# 그래프에서 사용할 색상을 통일합니다.
STROKE_COLORS = {
    "뇌졸중 없음": "#4C78A8",
    "뇌졸중 있음": "#E45756",
}


# ==================================================
# 3. 나이와 평균 혈당의 분포
# ==================================================
st.subheader("1. 나이와 평균 혈당의 분포")

st.write(
    "히스토그램은 값이 어느 구간에 얼마나 많이 분포하는지 보여 줍니다."
)

hist_col1, hist_col2 = st.columns(2)

with hist_col1:
    age_data = df[["age"]].dropna()

    fig_age_hist = px.histogram(
        age_data,
        x="age",
        nbins=30,
        title="나이 분포",
        labels={
            "age": "나이",
            "count": "사람 수",
        },
        color_discrete_sequence=["#4C78A8"],
    )

    fig_age_hist.update_layout(
        xaxis_title="나이",
        yaxis_title="사람 수",
        bargap=0.08,
    )

    st.plotly_chart(
        fig_age_hist,
        use_container_width=True,
    )

    st.caption(
        f"나이 결측치 제외: {len(age_data):,}명"
    )

with hist_col2:
    glucose_data = df[["avg_glucose_level"]].dropna()

    fig_glucose_hist = px.histogram(
        glucose_data,
        x="avg_glucose_level",
        nbins=30,
        title="평균 혈당 분포",
        labels={
            "avg_glucose_level": "평균 혈당",
            "count": "사람 수",
        },
        color_discrete_sequence=["#F2A541"],
    )

    fig_glucose_hist.update_layout(
        xaxis_title="평균 혈당",
        yaxis_title="사람 수",
        bargap=0.08,
    )

    st.plotly_chart(
        fig_glucose_hist,
        use_container_width=True,
    )

    st.caption(
        f"평균 혈당 결측치 제외: {len(glucose_data):,}명"
    )


# ==================================================
# 4. 뇌졸중 유무에 따른 나이와 평균 혈당 비교
# ==================================================
st.divider()
st.subheader("2. 뇌졸중 유무에 따른 분포 비교")

st.write(
    "상자그림은 중앙값, 사분위 범위, 이상치 등을 비교하는 데 도움이 됩니다. "
    "두 그룹의 분포가 어떻게 다른지 살펴보세요."
)

box_col1, box_col2 = st.columns(2)

with box_col1:
    age_box_data = df[
        ["age", "stroke_group"]
    ].dropna(subset=["age", "stroke_group"])

    fig_age_box = px.box(
        age_box_data,
        x="stroke_group",
        y="age",
        color="stroke_group",
        category_orders={
            "stroke_group": [
                "뇌졸중 없음",
                "뇌졸중 있음",
            ]
        },
        color_discrete_map=STROKE_COLORS,
        title="뇌졸중 유무에 따른 나이",
        labels={
            "stroke_group": "뇌졸중 여부",
            "age": "나이",
        },
        points="outliers",
    )

    fig_age_box.update_layout(
        showlegend=False,
        xaxis_title="뇌졸중 여부",
        yaxis_title="나이",
    )

    st.plotly_chart(
        fig_age_box,
        use_container_width=True,
    )

with box_col2:
    glucose_box_data = df[
        ["avg_glucose_level", "stroke_group"]
    ].dropna(subset=["avg_glucose_level", "stroke_group"])

    fig_glucose_box = px.box(
        glucose_box_data,
        x="stroke_group",
        y="avg_glucose_level",
        color="stroke_group",
        category_orders={
            "stroke_group": [
                "뇌졸중 없음",
                "뇌졸중 있음",
            ]
        },
        color_discrete_map=STROKE_COLORS,
        title="뇌졸중 유무에 따른 평균 혈당",
        labels={
            "stroke_group": "뇌졸중 여부",
            "avg_glucose_level": "평균 혈당",
        },
        points="outliers",
    )

    fig_glucose_box.update_layout(
        showlegend=False,
        xaxis_title="뇌졸중 여부",
        yaxis_title="평균 혈당",
    )

    st.plotly_chart(
        fig_glucose_box,
        use_container_width=True,
    )

# 그룹별 평균 계산
mean_data = df.dropna(subset=["stroke_group"])

group_means = (
    mean_data
    .groupby("stroke_group", observed=True)
    .agg(
        사람_수=("stroke", "size"),
        평균_나이=("age", "mean"),
        평균_혈당=("avg_glucose_level", "mean"),
    )
    .reindex(["뇌졸중 없음", "뇌졸중 있음"])
    .reset_index()
    .rename(columns={
        "stroke_group": "뇌졸중 여부",
        "사람_수": "사람 수",
        "평균_나이": "평균 나이",
        "평균_혈당": "평균 혈당",
    })
)

group_means["평균 나이"] = group_means["평균 나이"].round(2)
group_means["평균 혈당"] = group_means["평균 혈당"].round(2)

st.markdown("#### 뇌졸중 유무에 따른 평균 비교")

st.dataframe(
    group_means,
    hide_index=True,
    use_container_width=True,
)

st.caption(
    "평균은 각 변수의 결측치를 제외해 계산합니다. "
    "사람 수는 유효한 stroke 값이 있는 행을 기준으로 합니다."
)


# ==================================================
# 5. 고혈압과 심장병에 따른 뇌졸중 비율
# ==================================================
st.divider()
st.subheader("3. 고혈압과 심장병에 따른 뇌졸중 비율")

st.write(
    "각 건강 특성 그룹에서 stroke가 1인 사람의 비율을 비교합니다. "
    "비율은 각 그룹의 전체 사람 수를 기준으로 계산합니다."
)


def calculate_stroke_rate(data, feature, labels):
    """특성의 0/1 그룹별 인원과 뇌졸중 비율을 계산합니다."""
    valid = data[[feature, "stroke"]].dropna().copy()

    valid = valid[
        valid[feature].isin([0, 1])
        & valid["stroke"].isin([0, 1])
    ]

    result = (
        valid.groupby(feature, observed=True)
        .agg(
            사람_수=("stroke", "size"),
            뇌졸중_인원=("stroke", lambda values: (values == 1).sum()),
        )
        .reset_index()
    )

    result["뇌졸중_비율"] = (
        result["뇌졸중_인원"] / result["사람_수"] * 100
    )

    result["그룹"] = result[feature].map(labels)

    return result


risk_col1, risk_col2 = st.columns(2)

risk_specs = [
    {
        "column": "hypertension",
        "title": "고혈압 유무에 따른 뇌졸중 비율",
        "labels": {
            0: "고혈압 없음",
            1: "고혈압 있음",
        },
        "container": risk_col1,
    },
    {
        "column": "heart_disease",
        "title": "심장병 유무에 따른 뇌졸중 비율",
        "labels": {
            0: "심장병 없음",
            1: "심장병 있음",
        },
        "container": risk_col2,
    },
]

for spec in risk_specs:
    with spec["container"]:
        result = calculate_stroke_rate(
            df,
            spec["column"],
            spec["labels"],
        )

        st.markdown(f"#### {spec['title']}")

        if result.empty:
            st.warning("비율을 계산할 수 있는 데이터가 없습니다.")
            continue

        fig = px.bar(
            result,
            x="그룹",
            y="뇌졸중_비율",
            text=result["뇌졸중_비율"].map(
                lambda value: f"{value:.2f}%"
            ),
            custom_data=[
                "사람_수",
                "뇌졸중_인원",
            ],
            labels={
                "그룹": "그룹",
                "뇌졸중_비율": "뇌졸중 비율(%)",
            },
            color="그룹",
            color_discrete_sequence=["#4C78A8", "#E45756"],
        )

        fig.update_traces(
            textposition="outside",
            hovertemplate=(
                "%{x}<br>"
                "뇌졸중 비율: %{y:.2f}%<br>"
                "전체 인원: %{customdata[0]}명<br>"
                "뇌졸중 인원: %{customdata[1]}명"
                "<extra></extra>"
            ),
        )

        fig.update_layout(
            title=None,
            showlegend=False,
            xaxis_title="건강 특성",
            yaxis_title="뇌졸중 비율(%)",
            yaxis_ticksuffix="%",
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

        display_result = result[
            ["그룹", "사람_수", "뇌졸중_인원", "뇌졸중_비율"]
        ].rename(columns={
            "그룹": "그룹",
            "사람_수": "사람 수",
            "뇌졸중_인원": "뇌졸중 인원",
            "뇌졸중_비율": "뇌졸중 비율(%)",
        })

        display_result["뇌졸중 비율(%)"] = (
            display_result["뇌졸중 비율(%)"].round(2)
        )

        st.dataframe(
            display_result,
            hide_index=True,
            use_container_width=True,
        )


# ==================================================
# 6. BMI 결측자와 전체의 뇌졸중 비율 비교
# ==================================================
st.divider()
st.subheader("4. 체질량지수(bmi) 결측자 분석")

st.write(
    "BMI가 비어 있는 사람만 따로 확인하고, "
    "이 그룹의 뇌졸중 비율을 전체 데이터의 비율과 비교합니다."
)

valid_stroke = df["stroke"].isin([0, 1])
bmi_missing = df["bmi"].isna()

missing_bmi_group = df[bmi_missing & valid_stroke]
overall_group = df[valid_stroke]

missing_bmi_count = int(bmi_missing.sum())
missing_bmi_valid_count = len(missing_bmi_group)
missing_bmi_stroke_count = int(
    (missing_bmi_group["stroke"] == 1).sum()
)

overall_count = len(overall_group)
overall_stroke_count = int(
    (overall_group["stroke"] == 1).sum()
)

missing_bmi_rate = (
    missing_bmi_stroke_count / missing_bmi_valid_count * 100
    if missing_bmi_valid_count > 0
    else None
)

overall_rate = (
    overall_stroke_count / overall_count * 100
    if overall_count > 0
    else None
)

bmi_comparison = pd.DataFrame([
    {
        "대상": "BMI 결측자",
        "전체 인원": missing_bmi_count,
        "유효한 stroke 인원": missing_bmi_valid_count,
        "뇌졸중 인원": missing_bmi_stroke_count,
        "뇌졸중 비율(%)": (
            round(missing_bmi_rate, 2)
            if missing_bmi_rate is not None
            else None
        ),
    },
    {
        "대상": "전체 데이터",
        "전체 인원": len(df),
        "유효한 stroke 인원": overall_count,
        "뇌졸중 인원": overall_stroke_count,
        "뇌졸중 비율(%)": (
            round(overall_rate, 2)
            if overall_rate is not None
            else None
        ),
    },
])

st.dataframe(
    bmi_comparison,
    hide_index=True,
    use_container_width=True,
)

st.caption(
    "BMI 결측자 수는 bmi가 비어 있는 모든 행을 셉니다. "
    "뇌졸중 비율은 stroke가 0 또는 1로 확인되는 행만 사용합니다."
)


# ==================================================
# 7. 흡연 상태별 인원
# ==================================================
st.divider()
st.subheader("5. 흡연 상태별 사람 수")

st.write(
    "흡연 상태에 어떤 값이 있는지 확인하고, "
    "각 상태에 해당하는 사람 수를 살펴봅니다. "
    "빈 값도 별도의 범주로 표시합니다."
)

smoking = df["smoking_status"].astype("string")
smoking = smoking.fillna("결측값")

smoking_counts = (
    smoking
    .value_counts(dropna=False)
    .rename_axis("흡연 상태")
    .reset_index(name="사람 수")
)

st.dataframe(
    smoking_counts,
    hide_index=True,
    use_container_width=True,
)

st.caption(
    f"전체 행 수: {len(df):,}명"
)
