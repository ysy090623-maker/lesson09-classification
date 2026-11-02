import pandas as pd
import plotly.express as px
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
    page_title="분류 모델 | 뇌졸중 예측 실습실",
    page_icon="🧠",
    layout="wide",
)

DATA_URL = (
    "https://raw.githubusercontent.com/"
    "greatsong/modudata/main/data/stroke.csv"
)

KOREAN_NAMES = {
    "age": "나이",
    "avg_glucose_level": "평균 혈당",
    "bmi": "체질량지수",
    "hypertension": "고혈압",
    "heart_disease": "심장병",
}

FEATURES = [
    "age",
    "avg_glucose_level",
    "bmi",
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


st.title("🧠 분류 모델")
st.write(
    "뇌졸중 여부를 예측하는 두 분류 모델을 만들고, "
    "테스트 데이터에서 성능과 판단 과정을 살펴봅니다."
)


# --------------------------------------------------
# 입력 속성 선택
# --------------------------------------------------
st.subheader("1. 입력 속성 선택")

default_features = [
    "age",
    "avg_glucose_level",
    "hypertension",
    "heart_disease",
]

selected_features = st.multiselect(
    "모델 입력에 사용할 속성을 선택하세요.",
    options=FEATURES,
    default=default_features,
    format_func=lambda feature: KOREAN_NAMES[feature],
)

if len(selected_features) < 2:
    st.warning("입력 속성을 두 개 이상 선택해 주세요.")
    st.stop()

st.caption(
    "뇌졸중(stroke)이 1이면 양성(뇌졸중), 0이면 음성(뇌졸중 아님)입니다."
)


# --------------------------------------------------
# 고정된 학습·테스트 분할
# 번호 순으로 10명씩 묶어 각 묶음의 앞 3명을 테스트용으로 고정
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

# BMI가 선택된 경우에만 훈련 데이터의 중앙값으로 결측치를 채운다.
if "bmi" in selected_features:
    bmi_median = X_train["bmi"].median()
    X_train["bmi"] = X_train["bmi"].fillna(bmi_median)
    X_test["bmi"] = X_test["bmi"].fillna(bmi_median)

# 선택된 입력 속성에 남은 결측치가 있다면 훈련 데이터 중앙값으로 처리한다.
# 이 처리 역시 훈련 데이터만을 기준으로 한다.
train_medians = X_train.median(numeric_only=True)
X_train = X_train.fillna(train_medians)
X_test = X_test.fillna(train_medians)

st.caption(
    f"훈련 데이터 {len(train_df):,}명 · 테스트 데이터 {len(test_df):,}명"
)

if len(test_df) != 1533:
    st.warning(
        f"현재 데이터의 테스트 인원은 {len(test_df):,}명입니다. "
        "요청한 1,533명과 다르면 원본 데이터의 행 수를 확인해 주세요."
    )


# --------------------------------------------------
# 크기 맞추기: 훈련 데이터만으로 평균과 표준편차 계산
# --------------------------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# --------------------------------------------------
# 모델 학습
# --------------------------------------------------
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


# --------------------------------------------------
# 정확도
# --------------------------------------------------
st.subheader("2. 모델 정확도")

logistic_train_accuracy = accuracy_score(
    y_train,
    logistic_model.predict(X_train_scaled),
)
logistic_test_accuracy = accuracy_score(
    y_test,
    logistic_model.predict(X_test_scaled),
)

tree_train_accuracy = accuracy_score(
    y_train,
    tree_model.predict(X_train),
)
tree_test_accuracy = accuracy_score(
    y_test,
    tree_model.predict(X_test),
)

# 입력을 사용하지 않고 훈련 데이터의 다수 클래스로만 예측하는 기준 모델
majority_class = int(y_train.value_counts().idxmax())
baseline_train_accuracy = accuracy_score(
    y_train,
    [majority_class] * len(y_train),
)
baseline_test_accuracy = accuracy_score(
    y_test,
    [majority_class] * len(y_test),
)


def accuracy_card(title, test_accuracy, train_accuracy):
    st.markdown(
        f"""
        <div style="
            border: 1px solid rgba(128,128,128,0.35);
            border-radius: 12px;
            padding: 18px 20px;
            min-height: 145px;
        ">
            <div style="font-size: 1rem; margin-bottom: 10px;">
                {title}
            </div>
            <div style="font-size: 2.2rem; font-weight: 700;">
                {test_accuracy:.1%}
            </div>
            <div style="font-size: 0.85rem; opacity: 0.8; margin-top: 8px;">
                훈련 데이터 {train_accuracy:.1%}
                &nbsp;·&nbsp;
                테스트 데이터 {test_accuracy:.1%}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


card1, card2, card3 = st.columns(3)

with card1:
    accuracy_card(
        "로지스틱 회귀(확률로 답하는 모델)",
        logistic_test_accuracy,
        logistic_train_accuracy,
    )

with card2:
    accuracy_card(
        "의사결정트리(질문으로 답하는 모델)",
        tree_test_accuracy,
        tree_train_accuracy,
    )

with card3:
    accuracy_card(
        "입력 없이 훈련용 다수 클래스로만 답하는 모델",
        baseline_test_accuracy,
        baseline_train_accuracy,
    )


# --------------------------------------------------
# 2차원 그림에 사용할 두 속성 선택
# --------------------------------------------------
st.subheader("3. 테스트 데이터와 모델의 판단 경계")

axis_col1, axis_col2 = st.columns(2)

with axis_col1:
    x_feature = st.selectbox(
        "가로축",
        options=selected_features,
        index=0,
        format_func=lambda feature: KOREAN_NAMES[feature],
    )

with axis_col2:
    y_options = [feature for feature in selected_features if feature != x_feature]
    y_feature = st.selectbox(
        "세로축",
        options=y_options,
        index=0,
        format_func=lambda feature: KOREAN_NAMES[feature],
    )

# 두 축이 아닌 입력은 테스트 데이터의 중앙값에 고정한다.
fixed_values = {}
for feature in selected_features:
    if feature not in [x_feature, y_feature]:
        fixed_values[feature] = float(X_test[feature].median())

if fixed_values:
    fixed_text = " · ".join(
        f"{KOREAN_NAMES[feature]} = {value:.2f}"
        for feature, value in fixed_values.items()
    )
    st.write(f"두 축 이외의 속성은 테스트 데이터의 중앙값으로 고정했습니다: {fixed_text}")
else:
    st.write("선택한 입력 속성이 두 개뿐이므로 고정할 다른 속성은 없습니다.")


# --------------------------------------------------
# 2차원 평면에서 모델 예측을 계산
# --------------------------------------------------
x_min = float(X_test[x_feature].min())
x_max = float(X_test[x_feature].max())
y_min = float(X_test[y_feature].min())
y_max = float(X_test[y_feature].max())

x_padding = (x_max - x_min) * 0.08 or 1
y_padding = (y_max - y_min) * 0.08 or 1

x_grid = pd.Series(
    pd.np.linspace(x_min - x_padding, x_max + x_padding, 180)
)
y_grid = pd.Series(
    pd.np.linspace(y_min - y_padding, y_max + y_padding, 180)
)

grid_x, grid_y = pd.np.meshgrid(x_grid, y_grid)

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

grid_scaled = scaler.transform(grid_data)
grid_probabilities = logistic_model.predict_proba(grid_scaled)[:, 1]
grid_tree_predictions = tree_model.predict(grid_data)

fig = go.Figure()

# 의사결정트리가 나눈 칸을 먼저 옅은 색으로 표시한다.
fig.add_trace(
    go.Contour(
        x=x_grid,
        y=y_grid,
        z=grid_tree_predictions.reshape(grid_x.shape),
        colorscale=[
            [0, "rgba(100, 149, 237, 0.12)"],
            [1, "rgba(240, 128, 128, 0.12)"],
        ],
        showscale=False,
        contours=dict(
            start=0,
            end=1,
            size=1,
            coloring="fill",
            showlines=False,
        ),
        hoverinfo="skip",
        name="의사결정트리 구역",
    )
)

# 로지스틱 회귀의 양성 확률이 0.5가 되는 경계
fig.add_trace(
    go.Contour(
        x=x_grid,
        y=y_grid,
        z=grid_probabilities.reshape(grid_x.shape),
        contours=dict(
            start=0.5,
            end=0.5,
            size=1,
            coloring="lines",
            showlines=True,
        ),
        line=dict(color="black", width=3),
        showscale=False,
        hoverinfo="skip",
        name="로지스틱 회귀 0.5 경계",
    )
)

# 테스트 데이터의 실제 정답을 점의 색으로 표시
test_plot = test_df[selected_features + ["stroke"]].copy()
test_plot[x_feature] = X_test[x_feature].to_numpy()
test_plot[y_feature] = X_test[y_feature].to_numpy()
test_plot["실제 결과"] = test_plot["stroke"].map(
    {
        0: "뇌졸중 아님",
        1: "뇌졸중",
    }
)

fig.add_trace(
    go.Scatter(
        x=test_plot[x_feature],
        y=test_plot[y_feature],
        mode="markers",
        name="테스트 데이터",
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

fig.update_layout(
    title="테스트 데이터와 두 모델의 판단 영역",
    xaxis_title=KOREAN_NAMES[x_feature],
    yaxis_title=KOREAN_NAMES[y_feature],
    legend_title="표시",
)

st.plotly_chart(fig, use_container_width=True)

# 로지스틱 회귀의 경계가 현재 그림 범위에 있는지 확인
inside_boundary = (
    (grid_probabilities.min() <= 0.5)
    and (grid_probabilities.max() >= 0.5)
)

if not inside_boundary:
    st.info(
        "로지스틱 회귀의 0.5 경계가 현재 그림 범위 안에 나타나지 않습니다."
    )


# --------------------------------------------------
# 의사결정트리의 질문 가지 그림
# --------------------------------------------------
st.subheader("4. 의사결정트리가 던진 질문")

tree = tree_model.tree_
feature_names = [KOREAN_NAMES[feature] for feature in selected_features]

dot_lines = [
    "digraph Tree {",
    'graph [rankdir=TB, bgcolor="transparent"];',
    'node [shape=box, style="rounded,filled", fontname="Arial", fontsize=10];',
    'edge [fontname="Arial", fontsize=9];',
]

leaf_count = 0
negative_leaf_count = 0
asked_features = set()


def make_node_label(node_id):
    global leaf_count, negative_leaf_count

    samples = int(tree.n_node_samples[node_id])
    values = tree.value[node_id][0]
    negative_count = int(values[0])
    positive_count = int(values[1])
    positive_rate = positive_count / samples if samples else 0

    is_leaf = tree.children_left[node_id] == tree.children_right[node_id]

    if is_leaf:
        prediction = "뇌졸중" if positive_count > negative_count else "뇌졸중 아님"
        fill_color = "#F6C5C2" if positive_count > negative_count else "#C9DDF2"
        leaf_count += 1

        if prediction == "뇌졸중 아님":
            negative_leaf_count += 1

        return (
            f"답: {prediction}\\n"
            f"훈련 인원: {samples}명\\n"
            f"실제 뇌졸중: {positive_count}명\\n"
            f"뇌졸중 비율: {positive_rate:.1%}",
            fill_color,
        )

    feature_index = tree.feature[node_id]
    feature_name = selected_features[feature_index]
    asked_features.add(feature_name)
    threshold = tree.threshold[node_id]

    label = (
        f"{KOREAN_NAMES[feature_name]} ≤ {threshold:.2f}?\\n"
        f"훈련 인원: {samples}명\\n"
        f"실제 뇌졸중: {positive_count}명\\n"
        f"뇌졸중 비율: {positive_rate:.1%}"
    )

    return label, "#FFF2CC"


for node_id in range(tree.node_count):
    label, fill_color = make_node_label(node_id)
    escaped_label = label.replace('"', '\\"')
    dot_lines.append(
        f'{node_id} [label="{escaped_label}", fillcolor="{fill_color}"];'
    )

    left_child = tree.children_left[node_id]
    right_child = tree.children_right[node_id]

    if left_child != right_child:
        dot_lines.append(
            f'{node_id} -> {left_child} [label="예"];'
        )
        dot_lines.append(
            f'{node_id} -> {right_child} [label="아니요"];'
        )

dot_lines.append("}")
dot_string = "\n".join(dot_lines)

st.graphviz_chart(dot_string, use_container_width=True)

st.write(f"답을 내는 마디는 모두 {leaf_count}칸이며, 그중 {negative_leaf_count}칸은 ‘뇌졸중 아님’이라고 답합니다.")

asked_names = [
    KOREAN_NAMES[feature]
    for feature in selected_features
    if feature in asked_features
]

if asked_names:
    st.write("이 나무가 실제로 물은 속성: " + "·".join(asked_names))
else:
    st.write("이 나무는 입력 속성에 관한 질문을 나누지 않았습니다.")
