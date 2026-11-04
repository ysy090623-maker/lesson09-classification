import pandas as pd
import streamlit as st

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


# --------------------------------------------------
# 페이지 설정 및 데이터 불러오기
# --------------------------------------------------
st.set_page_config(
    page_title="채점표 | 뇌졸중 예측 실습실",
    page_icon="📋",
    layout="wide",
)

DATA_URL = (
    "https://raw.githubusercontent.com/"
    "greatsong/modudata/main/data/stroke.csv"
)

FEATURES = [
    "age",
    "avg_glucose_level",
    "bmi",
    "hypertension",
    "heart_disease",
]

KOREAN_NAMES = {
    "age": "나이",
    "avg_glucose_level": "평균 혈당",
    "bmi": "체질량지수",
    "hypertension": "고혈압",
    "heart_disease": "심장병",
}

DEFAULT_FEATURES = [
    "age",
    "avg_glucose_level",
    "hypertension",
    "heart_disease",
]

RANDOM_STATE = 42


@st.cache_data
def load_data():
    data = pd.read_csv(DATA_URL, encoding="utf-8")

    for column in FEATURES + ["stroke", "id"]:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    data = data.dropna(subset=["id", "stroke"])
    data = data.sort_values("id").reset_index(drop=True)

    return data


try:
    df = load_data()
except Exception as error:
    st.error("CSV 데이터를 불러오지 못했습니다. 인터넷 연결과 주소를 확인해 주세요.")
    st.caption(f"오류 내용: {error}")
    st.stop()


st.title("📋 채점표")
st.write(
    "테스트 데이터로 모델의 예측 결과를 채점하고, "
    "놓친 환자와 헛짚은 사람을 확인합니다."
)


# --------------------------------------------------
# 입력 속성 선택
# --------------------------------------------------
st.subheader("1. 입력 속성 선택")

selected_features = st.multiselect(
    "모델 입력에 사용할 속성을 선택하세요.",
    options=FEATURES,
    default=DEFAULT_FEATURES,
    format_func=lambda feature: KOREAN_NAMES[feature],
)

if len(selected_features) < 2:
    st.warning("입력 속성을 두 개 이상 선택해 주세요.")
    st.stop()

st.caption(
    "뇌졸중(stroke)이 1이면 양성(뇌졸중), 0이면 음성(뇌졸중 아님)입니다."
)


# --------------------------------------------------
# 분류 모델 페이지와 같은 방식으로 데이터 나누기
# 번호 순으로 정렬한 뒤 10명씩 묶어 각 묶음의 앞 3명을 테스트용으로 고정
# --------------------------------------------------
df = df.sort_values("id").reset_index(drop=True)
df["묶음 내 순서"] = df.index % 10

test_mask = df["묶음 내 순서"] < 3
train_df = df.loc[~test_mask].copy()
test_df = df.loc[test_mask].copy()

X_train = train_df[selected_features].copy()
X_test = test_df[selected_features].copy()

y_train = train_df["stroke"].astype(int)
y_test = test_df["stroke"].astype(int)

# BMI를 선택했을 때만 훈련 데이터의 중앙값으로 결측치를 채운다.
if "bmi" in selected_features:
    bmi_median = X_train["bmi"].median()
    X_train["bmi"] = X_train["bmi"].fillna(bmi_median)
    X_test["bmi"] = X_test["bmi"].fillna(bmi_median)

# 선택 속성에 다른 결측치가 있을 경우에도 훈련 데이터 기준으로 처리한다.
train_medians = X_train.median(numeric_only=True)
X_train = X_train.fillna(train_medians)
X_test = X_test.fillna(train_medians)

st.write(
    f"테스트용 사람 수: **{len(test_df):,}명** · "
    f"그중 실제 뇌졸중인 사람: **{int((y_test == 1).sum()):,}명**"
)

if len(test_df) != 1533:
    st.warning(
        f"현재 테스트 데이터는 {len(test_df):,}명입니다. "
        "요청한 1,533명과 다르면 원본 데이터의 행 수를 확인해 주세요."
    )


# --------------------------------------------------
# 크기 맞추기 및 모델 학습
# --------------------------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

logistic_model = LogisticRegression(
    max_iter=2000,
    random_state=RANDOM_STATE,
)
logistic_model.fit(X_train_scaled, y_train)

tree_model = DecisionTreeClassifier(
    max_depth=3,
    min_samples_leaf=5,
    random_state=RANDOM_STATE,
)
tree_model.fit(X_train, y_train)

logistic_predictions = logistic_model.predict(X_test_scaled)
tree_predictions = tree_model.predict(X_test)


# --------------------------------------------------
# 혼동행렬 및 지표 계산
# --------------------------------------------------
def get_confusion_counts(actual, predicted):
    """양성은 뇌졸중(1), 음성은 뇌졸중 아님(0)으로 계산한다."""
    actual = pd.Series(actual).astype(int)
    predicted = pd.Series(predicted).astype(int)

    tp = int(((actual == 1) & (predicted == 1)).sum())
    fp = int(((actual == 0) & (predicted == 1)).sum())
    fn = int(((actual == 1) & (predicted == 0)).sum())
    tn = int(((actual == 0) & (predicted == 0)).sum())

    return {
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn,
    }


def calculate_metrics(actual, predicted):
    counts = get_confusion_counts(actual, predicted)

    total = len(actual)
    actual_positive = counts["TP"] + counts["FN"]
    predicted_positive = counts["TP"] + counts["FP"]

    accuracy = (
        (counts["TP"] + counts["TN"]) / total
        if total
        else float("nan")
    )

    recall = (
        counts["TP"] / actual_positive
        if actual_positive
        else float("nan")
    )

    precision = (
        counts["TP"] / predicted_positive
        if predicted_positive
        else None
    )

    if precision is None or pd.isna(recall) or precision + recall == 0:
        f1 = None
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return {
        **counts,
        "정확도": accuracy,
        "재현율": recall,
        "정밀도": precision,
        "F1": f1,
    }


def metric_display(value):
    if value is None or pd.isna(value):
        return "계산할 수 없음"
    return f"{value:.3f}"


def confusion_table(counts):
    """행은 실제, 열은 예측이며 합계도 표시한다."""
    return pd.DataFrame(
        [
            [
                f'{counts["TN"]:,}명 (TN)',
                f'{counts["FP"]:,}명 (FP)',
                f'{counts["TN"] + counts["FP"]:,}명',
            ],
            [
                f'{counts["FN"]:,}명 (FN)',
                f'{counts["TP"]:,}명 (TP)',
                f'{counts["FN"] + counts["TP"]:,}명',
            ],
            [
                f'{counts["TN"] + counts["FN"]:,}명',
                f'{counts["FP"] + counts["TP"]:,}명',
                f'{counts["TN"] + counts["FP"] + counts["FN"] + counts["TP"]:,}명',
            ],
        ],
        index=["실제: 뇌졸중 아님", "실제: 뇌졸중", "합계"],
        columns=["예측: 뇌졸중 아님", "예측: 뇌졸중", "합계"],
    )


logistic_metrics = calculate_metrics(y_test, logistic_predictions)
tree_metrics = calculate_metrics(y_test, tree_predictions)

# 훈련 데이터에서 다수 클래스만 예측하는 기준 모델
majority_class = int(y_train.value_counts().idxmax())
baseline_predictions = [majority_class] * len(y_test)
baseline_metrics = calculate_metrics(y_test, baseline_predictions)


# --------------------------------------------------
# 혼동행렬
# --------------------------------------------------
st.subheader("2. 테스트 데이터 혼동행렬")
st.caption(
    "행은 실제 결과, 열은 모델의 예측 결과입니다. "
    "TP·FP·FN·TN은 각 칸의 사람 수와 함께 표시됩니다."
)

confusion_col1, confusion_col2 = st.columns(2)

with confusion_col1:
    st.markdown("**로지스틱 회귀(확률로 답하는 모델)**")
    st.dataframe(
        confusion_table(logistic_metrics),
        use_container_width=True,
    )

with confusion_col2:
    st.markdown("**의사결정트리(질문으로 답하는 모델)**")
    st.dataframe(
        confusion_table(tree_metrics),
        use_container_width=True,
    )


# --------------------------------------------------
# 정확도·재현율·정밀도·F1 비교
# --------------------------------------------------
st.subheader("3. 네 가지 지표 비교")

metric_rows = [
    {
        "모델": "로지스틱 회귀(확률로 답하는 모델)",
        "정확도": metric_display(logistic_metrics["정확도"]),
        "재현율": metric_display(logistic_metrics["재현율"]),
        "정밀도": metric_display(logistic_metrics["정밀도"]),
        "F1": metric_display(logistic_metrics["F1"]),
    },
    {
        "모델": "의사결정트리(질문으로 답하는 모델)",
        "정확도": metric_display(tree_metrics["정확도"]),
        "재현율": metric_display(tree_metrics["재현율"]),
        "정밀도": metric_display(tree_metrics["정밀도"]),
        "F1": metric_display(tree_metrics["F1"]),
    },
    {
        "모델": "훈련용 다수 클래스만 예측하는 모델",
        "정확도": metric_display(baseline_metrics["정확도"]),
        "재현율": metric_display(baseline_metrics["재현율"]),
        "정밀도": metric_display(baseline_metrics["정밀도"]),
        "F1": metric_display(baseline_metrics["F1"]),
    },
]

st.dataframe(
    pd.DataFrame(metric_rows),
    hide_index=True,
    use_container_width=True,
)


# --------------------------------------------------
# 놓친 환자와 헛짚은 사람 목록
# --------------------------------------------------
st.subheader("4. 놓친 환자와 헛짚은 사람")

model_choice = st.radio(
    "목록을 확인할 모델을 선택하세요.",
    options=[
        "로지스틱 회귀(확률로 답하는 모델)",
        "의사결정트리(질문으로 답하는 모델)",
    ],
    horizontal=True,
)

if model_choice == "로지스틱 회귀(확률로 답하는 모델)":
    chosen_predictions = logistic_predictions
    chosen_metrics = logistic_metrics
else:
    chosen_predictions = tree_predictions
    chosen_metrics = tree_metrics

# 테스트 행의 원본 정보와 예측 결과를 결합한다.
review_df = test_df[
    [
        "id",
        "age",
        "avg_glucose_level",
        "hypertension",
        "heart_disease",
        "stroke",
    ]
].copy()

review_df["예측"] = chosen_predictions

missed_df = review_df[
    (review_df["stroke"] == 1)
    & (review_df["예측"] == 0)
].copy()

false_alarm_df = review_df[
    (review_df["stroke"] == 0)
    & (review_df["예측"] == 1)
].copy()


def prepare_people_list(data):
    """표시용 열 이름과 값으로 정리하고 나이 내림차순으로 정렬한다."""
    result = data[
        [
            "id",
            "age",
            "avg_glucose_level",
            "hypertension",
            "heart_disease",
        ]
    ].copy()

    result = result.rename(
        columns={
            "id": "번호",
            "age": "나이",
            "avg_glucose_level": "평균 혈당",
            "hypertension": "고혈압 여부",
            "heart_disease": "심장병 여부",
        }
    )

    result["고혈압 여부"] = result["고혈압 여부"].map(
        {0: "없음", 1: "있음"}
    )
    result["심장병 여부"] = result["심장병 여부"].map(
        {0: "없음", 1: "있음"}
    )

    return result.sort_values(
        "나이",
        ascending=False,
        na_position="last",
    )


missed_list = prepare_people_list(missed_df)
false_alarm_list = prepare_people_list(false_alarm_df)

list_col1, list_col2 = st.columns(2)

with list_col1:
    st.markdown(f"**놓친 환자: {len(missed_list):,}명**")
    st.caption("실제 뇌졸중인데 모델이 뇌졸중이 아니라고 예측한 사람")

    st.dataframe(
        missed_list.head(20),
        hide_index=True,
        use_container_width=True,
    )

    if len(missed_list) > 20:
        st.caption("목록은 나이가 많은 순서로 앞 20명만 표시했습니다.")

with list_col2:
    st.markdown(f"**헛짚은 사람: {len(false_alarm_list):,}명**")
    st.caption("실제 뇌졸중이 아닌데 모델이 뇌졸중이라고 예측한 사람")

    st.dataframe(
        false_alarm_list.head(20),
        hide_index=True,
        use_container_width=True,
    )

    if len(false_alarm_list) > 20:
        st.caption("목록은 나이가 많은 순서로 앞 20명만 표시했습니다.")

st.write(
    f"선택한 모델의 놓친 환자 수는 {len(missed_list):,}명으로 "
    f"혼동행렬의 FN과 같고, 헛짚은 사람 수는 {len(false_alarm_list):,}명으로 "
    f"혼동행렬의 FP와 같습니다."
)
