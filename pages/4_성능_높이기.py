import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


# --------------------------------------------------
# 페이지 설정 및 데이터
# --------------------------------------------------
st.set_page_config(
    page_title="성능 높이기 | 뇌졸중 예측 실습실",
    page_icon="⚖️",
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


st.title("⚖️ 성능 높이기")
st.write(
    "뇌졸중을 겪은 사람이 적을 때, 양쪽의 가중치를 맞추기 전과 후의 "
    "예측 결과를 같은 테스트 데이터에서 비교합니다."
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

st.caption("뇌졸중(stroke)이 1이면 양성(뇌졸중), 0이면 음성(뇌졸중 아님)입니다.")


# --------------------------------------------------
# 채점표 페이지와 같은 데이터 분할
# 번호 순 정렬 후 10명마다 앞 3명을 테스트 데이터로 지정
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

# BMI가 선택된 경우에만 훈련 데이터 중앙값으로 결측치를 채운다.
if "bmi" in selected_features:
    bmi_median = X_train["bmi"].median()
    X_train["bmi"] = X_train["bmi"].fillna(bmi_median)
    X_test["bmi"] = X_test["bmi"].fillna(bmi_median)

# 선택된 속성에 다른 결측치가 있으면 훈련 데이터 중앙값으로 처리한다.
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
# 크기 맞추기: 훈련 데이터만으로 평균과 표준편차 계산
# --------------------------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# --------------------------------------------------
# 두 모델: 가중치 맞추기 전과 후
# --------------------------------------------------
logistic_before = LogisticRegression(
    max_iter=2000,
    random_state=RANDOM_STATE,
)
logistic_after = LogisticRegression(
    max_iter=2000,
    class_weight="balanced",
    random_state=RANDOM_STATE,
)

tree_before = DecisionTreeClassifier(
    max_depth=3,
    min_samples_leaf=5,
    random_state=RANDOM_STATE,
)
tree_after = DecisionTreeClassifier(
    max_depth=3,
    min_samples_leaf=5,
    class_weight="balanced",
    random_state=RANDOM_STATE,
)

logistic_before.fit(X_train_scaled, y_train)
logistic_after.fit(X_train_scaled, y_train)
tree_before.fit(X_train, y_train)
tree_after.fit(X_train, y_train)

predictions = {
    "로지스틱 회귀(확률로 답하는 모델)": {
        "전": logistic_before.predict(X_test_scaled),
        "후": logistic_after.predict(X_test_scaled),
    },
    "의사결정트리(질문으로 답하는 모델)": {
        "전": tree_before.predict(X_test),
        "후": tree_after.predict(X_test),
    },
}


# --------------------------------------------------
# 혼동행렬과 지표 계산
# --------------------------------------------------
def get_counts(actual, predicted):
    actual = np.asarray(actual, dtype=int)
    predicted = np.asarray(predicted, dtype=int)

    tp = int(((actual == 1) & (predicted == 1)).sum())
    fp = int(((actual == 0) & (predicted == 1)).sum())
    fn = int(((actual == 1) & (predicted == 0)).sum())
    tn = int(((actual == 0) & (predicted == 0)).sum())

    return {"TP": tp, "FP": fp, "FN": fn, "TN": tn}


def calculate_metrics(actual, predicted):
    counts = get_counts(actual, predicted)

    total = len(actual)
    actual_positive = counts["TP"] + counts["FN"]
    predicted_positive = counts["TP"] + counts["FP"]

    accuracy = (
        (counts["TP"] + counts["TN"]) / total
        if total
        else np.nan
    )

    recall = (
        counts["TP"] / actual_positive
        if actual_positive
        else np.nan
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


def show_metric(value):
    if value is None or pd.isna(value):
        return "계산할 수 없음"
    return f"{value:.3f}"


all_metrics = {
    model_name: {
        period: calculate_metrics(y_test, predicted)
        for period, predicted in periods.items()
    }
    for model_name, periods in predictions.items()
}


# --------------------------------------------------
# 전후 비교 표
# --------------------------------------------------
st.subheader("2. 가중치를 맞추기 전과 후의 채점 결과")

for model_name, periods in all_metrics.items():
    st.markdown(f"**{model_name}**")

    before = periods["전"]
    after = periods["후"]

    comparison = pd.DataFrame(
        [
            {
                "구분": "TP",
                "가중치 맞추기 전": before["TP"],
                "가중치 맞춘 뒤": after["TP"],
            },
            {
                "구분": "FP",
                "가중치 맞추기 전": before["FP"],
                "가중치 맞춘 뒤": after["FP"],
            },
            {
                "구분": "FN",
                "가중치 맞추기 전": before["FN"],
                "가중치 맞춘 뒤": after["FN"],
            },
            {
                "구분": "TN",
                "가중치 맞추기 전": before["TN"],
                "가중치 맞춘 뒤": after["TN"],
            },
            {
                "구분": "정확도",
                "가중치 맞추기 전": show_metric(before["정확도"]),
                "가중치 맞춘 뒤": show_metric(after["정확도"]),
            },
            {
                "구분": "재현율",
                "가중치 맞추기 전": show_metric(before["재현율"]),
                "가중치 맞춘 뒤": show_metric(after["재현율"]),
            },
            {
                "구분": "정밀도",
                "가중치 맞추기 전": show_metric(before["정밀도"]),
                "가중치 맞춘 뒤": show_metric(after["정밀도"]),
            },
            {
                "구분": "F1",
                "가중치 맞추기 전": show_metric(before["F1"]),
                "가중치 맞춘 뒤": show_metric(after["F1"]),
            },
        ]
    )

    st.dataframe(
        comparison,
        hide_index=True,
        use_container_width=True,
    )

    st.write(
        f"찾아낸 환자(TP): {before['TP']}명 → {after['TP']}명 · "
        f"놓친 환자(FN): {before['FN']}명 → {after['FN']}명 · "
        f"헛짚은 사람(FP): {before['FP']}명 → {after['FP']}명"
    )


# --------------------------------------------------
# 경계면 시각화용 속성 선택
# --------------------------------------------------
st.subheader("3. 가중치 전후의 0.5 경계 비교")

if len(selected_features) == 2:
    x_feature = selected_features[0]
    y_feature = selected_features[1]
    z_feature = None
else:
    axis_col1, axis_col2, axis_col3 = st.columns(3)

    with axis_col1:
        x_feature = st.selectbox(
            "가로축",
            options=selected_features,
            index=0,
            format_func=lambda feature: KOREAN_NAMES[feature],
        )

    remaining_y = [feature for feature in selected_features if feature != x_feature]

    with axis_col2:
        y_feature = st.selectbox(
            "세로축",
            options=remaining_y,
            index=0,
            format_func=lambda feature: KOREAN_NAMES[feature],
        )

    remaining_z = [
        feature
        for feature in selected_features
        if feature not in [x_feature, y_feature]
    ]

    with axis_col3:
        z_feature = st.selectbox(
            "깊이축",
            options=remaining_z,
            index=0,
            format_func=lambda feature: KOREAN_NAMES[feature],
        )


# 축으로 선택하지 않은 입력 속성은 테스트 데이터 중앙값으로 고정
axis_features = [x_feature, y_feature]
if z_feature is not None:
    axis_features.append(z_feature)

fixed_values = {
    feature: float(X_test[feature].median())
    for feature in selected_features
    if feature not in axis_features
}

if fixed_values:
    fixed_text = " · ".join(
        f"{KOREAN_NAMES[feature]} = {value:.2f}"
        for feature, value in fixed_values.items()
    )
    st.write(f"축으로 고르지 않은 속성은 테스트 데이터 중앙값으로 고정했습니다: {fixed_text}")
else:
    st.write("선택한 모든 속성이 축에 사용되므로 고정할 다른 속성은 없습니다.")


# --------------------------------------------------
# 2차원 산점도와 로지스틱 회귀 경계선
# --------------------------------------------------
def make_grid_data(x_values, y_values, z_values=None):
    if z_values is None:
        grid_x, grid_y = np.meshgrid(x_values, y_values)

        grid_data = pd.DataFrame(
            {
                feature: (
                    grid_x.ravel()
                    if feature == x_feature
                    else grid_y.ravel()
                    if feature == y_feature
                    else fixed_values[feature]
                )
                for feature in selected_features
            }
        )
        return grid_x, grid_y, None, grid_data

    grid_x, grid_y, grid_z = np.meshgrid(
        x_values,
        y_values,
        z_values,
        indexing="xy",
    )

    grid_data = pd.DataFrame(
        {
            feature: (
                grid_x.ravel()
                if feature == x_feature
                else grid_y.ravel()
                if feature == y_feature
                else grid_z.ravel()
                if feature == z_feature
                else fixed_values[feature]
            )
            for feature in selected_features
        }
    )
    return grid_x, grid_y, grid_z, grid_data


def make_axis_range(series, points=80):
    low = float(series.min())
    high = float(series.max())
    padding = (high - low) * 0.08
    if padding == 0:
        padding = 1
    return np.linspace(low - padding, high + padding, points)


if z_feature is None:
    x_values = make_axis_range(X_test[x_feature], 180)
    y_values = make_axis_range(X_test[y_feature], 180)

    grid_x, grid_y, _, grid_data = make_grid_data(
        x_values,
        y_values,
    )

    grid_scaled = scaler.transform(grid_data)
    probability_before = logistic_before.predict_proba(grid_scaled)[:, 1]
    probability_after = logistic_after.predict_proba(grid_scaled)[:, 1]

    fig = go.Figure()

    # 실제 테스트 데이터를 색으로 구분
    test_plot = test_df[
        ["id", "stroke"]
    ].copy()
    test_plot[x_feature] = X_test[x_feature].to_numpy()
    test_plot[y_feature] = X_test[y_feature].to_numpy()
    test_plot["실제 결과"] = test_plot["stroke"].map(
        {0: "뇌졸중 아님", 1: "뇌졸중"}
    )

    fig.add_trace(
        go.Scatter(
            x=test_plot[x_feature],
            y=test_plot[y_feature],
            mode="markers",
            name="테스트 데이터(실제 결과)",
            marker=dict(
                size=7,
                color=test_plot["stroke"],
                colorscale=[
                    [0, "#2878B5"],
                    [1, "#D94841"],
                ],
                cmin=0,
                cmax=1,
                line=dict(width=0.5, color="white"),
                showscale=False,
            ),
            customdata=test_plot[["실제 결과"]].to_numpy(),
            hovertemplate=(
                f"{KOREAN_NAMES[x_feature]}: %{{x:.2f}}<br>"
                f"{KOREAN_NAMES[y_feature]}: %{{y:.2f}}<br>"
                "실제 결과: %{customdata[0]}<extra></extra>"
            ),
        )
    )

    # 전·후 확률 0.5 경계선을 겹쳐 표시
    for probability, label, color in [
        (probability_before, "가중치 맞추기 전 경계(0.5)", "#2878B5"),
        (probability_after, "가중치 맞춘 뒤 경계(0.5)", "#E67E22"),
    ]:
        fig.add_trace(
            go.Contour(
                x=x_values,
                y=y_values,
                z=probability.reshape(grid_x.shape),
                contours=dict(
                    start=0.5,
                    end=0.5,
                    size=1,
                    coloring="lines",
                    showlines=True,
                ),
                line=dict(color=color, width=3),
                showscale=False,
                hoverinfo="skip",
                name=label,
            )
        )

    fig.update_layout(
        title="테스트 데이터와 로지스틱 회귀의 0.5 경계",
        xaxis_title=KOREAN_NAMES[x_feature],
        yaxis_title=KOREAN_NAMES[y_feature],
        legend_title="표시",
    )

    st.plotly_chart(fig, use_container_width=True)

    for probability, label in [
        (probability_before, "가중치 맞추기 전"),
        (probability_after, "가중치 맞춘 뒤"),
    ]:
        if probability.min() > 0.5 or probability.max() < 0.5:
            st.info(f"{label}의 0.5 경계선은 현재 그림 범위 밖에 있습니다.")


# --------------------------------------------------
# 3차원 산점도와 전후 0.5 평면
# --------------------------------------------------
else:
    x_values = make_axis_range(X_test[x_feature], 35)
    y_values = make_axis_range(X_test[y_feature], 35)
    z_values = make_axis_range(X_test[z_feature], 35)

    grid_x, grid_y, grid_z, grid_data = make_grid_data(
        x_values,
        y_values,
        z_values,
    )

    grid_scaled = scaler.transform(grid_data)
    probability_before = logistic_before.predict_proba(grid_scaled)[:, 1]
    probability_after = logistic_after.predict_proba(grid_scaled)[:, 1]

    # 격자에서 확률 0.5인 점을 표면으로 표현
    def boundary_surface(probabilities):
        probability_grid = probabilities.reshape(grid_x.shape)
        surface_x = []
        surface_y = []
        surface_z = []

        for yi in range(probability_grid.shape[1]):
            column = probability_grid[:, yi, :]
            for zi in range(column.shape[1]):
                values = column[:, zi]
                crossing = np.where(
                    (values[:-1] - 0.5) * (values[1:] - 0.5) <= 0
                )[0]

                for xi in crossing:
                    p1 = values[xi]
                    p2 = values[xi + 1]

                    if p1 == p2:
                        fraction = 0.5
                    else:
                        fraction = (0.5 - p1) / (p2 - p1)

                    surface_x.append(
                        x_values[xi]
                        + fraction * (x_values[xi + 1] - x_values[xi])
                    )
                    surface_y.append(y_values[yi])
                    surface_z.append(z_values[zi])

        if not surface_x:
            return None

        return surface_x, surface_y, surface_z

    fig3d = go.Figure()

    test_plot = test_df[["stroke"]].copy()
    test_plot[x_feature] = X_test[x_feature].to_numpy()
    test_plot[y_feature] = X_test[y_feature].to_numpy()
    test_plot[z_feature] = X_test[z_feature].to_numpy()
    test_plot["실제 결과"] = test_plot["stroke"].map(
        {0: "뇌졸중 아님", 1: "뇌졸중"}
    )

    fig3d.add_trace(
        go.Scatter3d(
            x=test_plot[x_feature],
            y=test_plot[y_feature],
            z=test_plot[z_feature],
            mode="markers",
            name="테스트 데이터(실제 결과)",
            marker=dict(
                size=4,
                color=test_plot["stroke"],
                colorscale=[
                    [0, "#2878B5"],
                    [1, "#D94841"],
                ],
                cmin=0,
                cmax=1,
                line=dict(width=0.5, color="white"),
                showscale=False,
            ),
            customdata=test_plot[["실제 결과"]].to_numpy(),
            hovertemplate=(
                f"{KOREAN_NAMES[x_feature]}: %{{x:.2f}}<br>"
                f"{KOREAN_NAMES[y_feature]}: %{{y:.2f}}<br>"
                f"{KOREAN_NAMES[z_feature]}: %{{z:.2f}}<br>"
                "실제 결과: %{customdata[0]}<extra></extra>"
            ),
        )
    )

    # 3차원 경계는 확률 0.5를 가로지르는 격자 점들을 표면으로 표시
    for probabilities, label, color in [
        (probability_before, "가중치 맞추기 전 경계(0.5)", "#2878B5"),
        (probability_after, "가중치 맞춘 뒤 경계(0.5)", "#E67E22"),
    ]:
        surface = boundary_surface(probabilities)

        if surface is None:
            st.info(f"{label}는 현재 3차원 그림 범위 밖에 있습니다.")
            continue

        sx, sy, sz = surface
        fig3d.add_trace(
            go.Mesh3d(
                x=sx,
                y=sy,
                z=sz,
                color=color,
                opacity=0.25,
                alphahull=0,
                name=label,
                hoverinfo="skip",
            )
        )

    fig3d.update_layout(
        title="3차원 테스트 데이터와 로지스틱 회귀의 0.5 경계면",
        scene=dict(
            xaxis_title=KOREAN_NAMES[x_feature],
            yaxis_title=KOREAN_NAMES[y_feature],
            zaxis_title=KOREAN_NAMES[z_feature],
        ),
        legend_title="표시",
    )

    st.plotly_chart(fig3d, use_container_width=True)

    if (
        probability_before.min() > 0.5
        or probability_before.max() < 0.5
    ):
        st.info("가중치 맞추기 전의 0.5 경계면은 현재 3차원 그림 범위 밖에 있습니다.")

    if (
        probability_after.min() > 0.5
        or probability_after.max() < 0.5
    ):
        st.info("가중치 맞춘 뒤의 0.5 경계면은 현재 3차원 그림 범위 밖에 있습니다.")
