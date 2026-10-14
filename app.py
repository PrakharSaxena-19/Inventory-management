


import os
import warnings
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import streamlit as st


from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

warnings.filterwarnings("ignore")

# 1. Look for this line near the top of your file:
st.set_page_config(
    page_title="Stock Risk Intelligence",
    layout="wide"
)

# 2. PASTE THIS SNIPPET DIRECTLY BELOW IT:
st.markdown("""
    <style>
        /* Hide the header and colored top line */
        header[data-testid="stHeader"] {
            display: none !important;
        }
        /* Adjust top padding so title has breathing room */
        .block-container {
            padding-top: 2rem !important;
        }
    </style>
""", unsafe_allow_html=True)

# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Stock Risk Intelligence",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# 2. PROFESSIONAL CSS DESIGN
# ============================================================

st.markdown("""
<style>
    .stApp {
        background-color: #F5F7FB;
    }

    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 2rem;
    }

    .main-title {
        font-size: 34px;
        font-weight: 800;
        color: #152238;
        margin-bottom: 3px;
    }

    .subtitle {
        color: #637083;
        font-size: 15px;
        margin-bottom: 22px;
    }

    .section-title {
        font-size: 23px;
        font-weight: 750;
        color: #152238;
        margin-top: 15px;
        margin-bottom: 12px;
    }

    .metric-card {
        background: #FFFFFF;
        padding: 20px;
        border-radius: 13px;
        border: 1px solid #E3E8F0;
        box-shadow: 0 3px 12px rgba(25, 40, 70, 0.04);
        min-height: 125px;
    }

    .metric-label {
        color: #65748B;
        font-size: 14px;
        font-weight: 600;
    }

    .metric-value {
        color: #14243A;
        font-size: 28px;
        font-weight: 800;
        margin-top: 9px;
    }

    .metric-note {
        color: #758198;
        font-size: 12px;
        margin-top: 4px;
    }

    .info-card {
        background: white;
        padding: 18px;
        border-radius: 12px;
        border: 1px solid #E3E8F0;
        margin-bottom: 12px;
    }

    .risk-high {
        background-color: #FDECEC;
        color: #B42318;
        border-left: 5px solid #D92D20;
        padding: 18px;
        border-radius: 9px;
        font-weight: 650;
        margin-top: 12px;
        margin-bottom: 12px;
    }

    .risk-low {
        background-color: #EAF7EF;
        color: #187344;
        border-left: 5px solid #219653;
        padding: 18px;
        border-radius: 9px;
        font-weight: 650;
        margin-top: 12px;
        margin-bottom: 12px;
    }

    div[data-testid="stMetric"] {
        background: white;
        padding: 15px;
        border-radius: 12px;
        border: 1px solid #E3E8F0;
    }

    div[data-testid="stForm"] {
        background: white;
        padding: 20px;
        border-radius: 14px;
        border: 1px solid #E3E8F0;
    }

    div.stButton > button,
    div.stFormSubmitButton > button {
        background-color: #2457D6;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 10px 18px;
        font-weight: 650;
        min-height: 43px;
    }

    div.stButton > button:hover,
    div.stFormSubmitButton > button:hover {
        background-color: #173FA8;
        color: white;
    }

    [data-testid="stSidebar"] {
        background-color: #FFFFFF;
        border-right: 1px solid #E3E8F0;
    }

    #MainMenu, footer {
        visibility: hidden;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# 3. FILE PATHS AND FEATURE DEFINITIONS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(
    BASE_DIR,
    "stock_risk_prediction_dataset (1).xlsx"
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "artifacts",
    "stock_risk_model.joblib"
)

NUMERIC_FEATURES = [
    "current_stock",
    "avg_daily_demand",
    "lead_time_days",
    "reorder_point",
    "sales_last_30_days",
    "stock_turnover_ratio",
    "forecast_error"
]

CATEGORICAL_FEATURES = [
    "season",
    "item_category",
    "supplier_reliability"
]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "stock_status"

MODEL_OPTIONS = [
    "Logistic Regression",
    "Random Forest",
    "XGBoost"
]

# ============================================================
# 4. LOAD DATASET
# ============================================================

@st.cache_data(show_spinner=False)
def load_data():
    if not os.path.exists(DATA_PATH):
        return None

    data = pd.read_excel(DATA_PATH)

    data.columns = data.columns.str.strip().str.lower()

    return data


# ============================================================
# 5. LOAD SAVED MODEL BUNDLE
# ============================================================

@st.cache_resource(show_spinner=False)
def load_saved_bundle():
    if not os.path.exists(MODEL_PATH):
        return None

    return joblib.load(MODEL_PATH)


# ============================================================
# 6. MODEL TRAINING AND MODEL SELECTION
# ============================================================

@st.cache_resource(show_spinner="Preparing the three machine learning models...")
def prepare_models(data):
    required_columns = FEATURES + [TARGET]

    missing_columns = [
        column for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing required dataset columns: "
            + ", ".join(missing_columns)
        )

    clean_data = data[required_columns].copy()

    clean_data[TARGET] = (
        clean_data[TARGET]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    clean_data = clean_data.replace(
        [np.inf, -np.inf],
        np.nan
    )

    clean_data = clean_data.dropna(subset=[TARGET])

    X = clean_data[FEATURES].copy()
    y_text = clean_data[TARGET].copy()

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_text)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent")
            ),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore")
            )
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_transformer,
                NUMERIC_FEATURES
            ),
            (
                "categorical",
                categorical_transformer,
                CATEGORICAL_FEATURES
            )
        ]
    )

    models = {
        "Logistic Regression": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=1000,
                        class_weight="balanced",
                        random_state=42
                    )
                )
            ]
        ),

        "Random Forest": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=150,
                        max_depth=15,
                        min_samples_split=5,
                        class_weight="balanced",
                        random_state=42,
                        n_jobs=-1
                    )
                )
            ]
        )
    }

    # Reuse the saved XGBoost model when available.
    saved_bundle = load_saved_bundle()

    if (
        isinstance(saved_bundle, dict)
        and "model" in saved_bundle
        and saved_bundle["model"] is not None
    ):
        models["XGBoost"] = saved_bundle["model"]

    else:
        try:
            from xgboost import XGBClassifier

            models["XGBoost"] = Pipeline(
                steps=[
                    ("preprocessor", preprocessor),
                    (
                        "classifier",
                        XGBClassifier(
                            n_estimators=150,
                            max_depth=6,
                            learning_rate=0.08,
                            subsample=0.9,
                            colsample_bytree=0.9,
                            objective="multi:softprob"
                            if len(label_encoder.classes_) > 2
                            else "binary:logistic",
                            eval_metric="mlogloss",
                            random_state=42,
                            n_jobs=-1
                        )
                    )
                ]
            )

        except ImportError:
            models["XGBoost"] = None

    for model_name, model in models.items():
        if model is None:
            continue

        # Fit only the alternative models created in this app.
        if model_name != "XGBoost" or not (
            isinstance(saved_bundle, dict)
            and "model" in saved_bundle
            and saved_bundle["model"] is model
        ):
            model.fit(X_train, y_train)

    return {
        "models": models,
        "label_encoder": label_encoder,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "data": clean_data
    }


# ============================================================
# 7. COMMON HELPER FUNCTIONS
# ============================================================

def metric_card(label, value, note=""):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def format_label(label):
    return str(label).replace("_", " ").title()


def get_prediction(model, input_data, label_encoder):
    prediction = model.predict(input_data)
    prediction = int(np.asarray(prediction).ravel()[0])

    classes = list(label_encoder.classes_)

    if 0 <= prediction < len(classes):
        return classes[prediction]

    return str(prediction)


def get_probabilities(model, input_data, label_encoder):
    if not hasattr(model, "predict_proba"):
        return None

    probabilities = model.predict_proba(input_data)[0]

    model_classes = getattr(
        model,
        "classes_",
        None
    )

    if model_classes is None and hasattr(model, "named_steps"):
        classifier = model.named_steps.get("classifier")
        if classifier is not None:
            model_classes = getattr(classifier, "classes_", None)

    if model_classes is None:
        return None

    output = {}

    for class_id, probability in zip(
        model_classes,
        probabilities
    ):
        class_id = int(class_id)

        if 0 <= class_id < len(label_encoder.classes_):
            label = label_encoder.classes_[class_id]
            output[label] = float(probability)

    return output


def get_model_metrics(model, X_test, y_test):
    predictions = model.predict(X_test)

    return {
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(
            y_test,
            predictions,
            average="weighted",
            zero_division=0
        ),
        "Recall": recall_score(
            y_test,
            predictions,
            average="weighted",
            zero_division=0
        ),
        "F1 Score": f1_score(
            y_test,
            predictions,
            average="weighted",
            zero_division=0
        ),
        "Predictions": predictions
    }


def make_recommendations(
    risk,
    current_stock,
    avg_daily_demand,
    lead_time_days,
    reorder_point,
    supplier_reliability
):
    recommendations = []

    expected_lead_time_demand = (
        max(avg_daily_demand, 0) * max(lead_time_days, 0)
    )

    if risk == "stockout":
        recommendations.append(
            "Prioritize replenishment because the model identifies "
            "a potential stockout risk."
        )

        if current_stock <= reorder_point:
            recommendations.append(
                "Current stock is at or below the reorder point. "
                "Review a purchase order as soon as possible."
            )

        if current_stock < expected_lead_time_demand:
            recommendations.append(
                "Available inventory may not cover estimated demand "
                "during supplier lead time."
            )

        if supplier_reliability == "low":
            recommendations.append(
                "Confirm supplier availability and delivery dates. "
                "Consider an approved backup supplier."
            )

        recommendations.append(
            "Check recent sales, demand forecasts, and open orders "
            "before finalizing replenishment quantity."
        )

    elif risk == "overstock":
        recommendations.append(
            "Review excess inventory and identify slow-moving items."
        )

        recommendations.append(
            "Consider reducing future orders until stock approaches "
            "the target inventory level."
        )

        recommendations.append(
            "Review demand forecasts and sales promotions before "
            "making purchasing decisions."
        )

        if current_stock > 0 and avg_daily_demand > 0:
            estimated_days_cover = current_stock / avg_daily_demand

            if estimated_days_cover > max(lead_time_days * 3, 30):
                recommendations.append(
                    f"Estimated stock coverage is approximately "
                    f"{estimated_days_cover:.1f} days. Review the "
                    "inventory holding cost and expiry risk where relevant."
                )

    else:
        recommendations.append(
            "Continue routine inventory monitoring and review demand "
            "and stock levels regularly."
        )

    return recommendations


# ============================================================
# 8. LOAD APPLICATION DATA
# ============================================================

data = load_data()

if data is None:
    st.error(
        "Dataset not found. Place "
        "'stock_risk_prediction_dataset (1).xlsx' "
        "in the same folder as app.py."
    )
    st.stop()

try:
    application = prepare_models(data)

except Exception as error:
    st.error("Unable to prepare the application models.")
    st.exception(error)
    st.stop()

models = application["models"]
label_encoder = application["label_encoder"]
X_test = application["X_test"]
y_test = application["y_test"]
clean_data = application["data"]

# ============================================================
# 9. HEADER
# ============================================================

st.markdown(
    '<div class="main-title">📦 Stock Risk Intelligence</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="subtitle">
        AI-powered inventory risk monitoring, predictive analytics,
        model comparison, and inventory decision support.
    </div>
    """,
    unsafe_allow_html=True
)

# ============================================================
# 10. TOP NAVIGATION
# ============================================================

st.markdown("### Navigation")

page = st.radio(
    "Select a page",
    [
        "Dashboard",
        "Predict Stock Risk",
        "Model Performance",
        "Business Recommendations"
    ],
    horizontal=True,
    label_visibility="collapsed"
)

st.divider()

# ============================================================
# 11. SIDEBAR MODEL SELECTOR
# ============================================================

st.sidebar.markdown("## ⚙️ Model Settings")

available_models = [
    name for name in MODEL_OPTIONS
    if models.get(name) is not None
]

if not available_models:
    st.sidebar.error(
        "No trained models are available. Check your dataset "
        "and saved model files."
    )
    st.stop()

selected_model_name = st.sidebar.selectbox(
    "Select Machine Learning Model",
    available_models,
    index=(
        available_models.index("XGBoost")
        if "XGBoost" in available_models
        else 0
    )
)

selected_model = models[selected_model_name]

st.sidebar.caption(
    "Choose a model to use for stock-risk prediction and evaluation."
)

st.sidebar.divider()

st.sidebar.markdown("### Project Information")
st.sidebar.write("**Domain:** Supply Chain Analytics")
st.sidebar.write("**Task:** Stock Risk Classification")
st.sidebar.write("**Target:** Stock Status")
st.sidebar.write("**Models:** Logistic Regression, Random Forest, XGBoost")

st.sidebar.caption(
    "Predictions support inventory decisions; verify them against "
    "actual stock records and business requirements."
)

# ============================================================
# 12. MODEL METRICS
# ============================================================

with st.spinner(f"Evaluating {selected_model_name}..."):
    try:
        selected_metrics = get_model_metrics(
            selected_model,
            X_test,
            y_test
        )
    except Exception as error:
        st.error(
            f"Could not evaluate {selected_model_name}. "
            "Check whether the saved model matches the current dataset."
        )
        st.exception(error)
        st.stop()

# ============================================================
# 13. DASHBOARD
# ============================================================

if page == "Dashboard":

    st.markdown(
        '<div class="section-title">Inventory Overview</div>',
        unsafe_allow_html=True
    )

    total_records = len(clean_data)
    total_stock = clean_data["current_stock"].sum()
    avg_stock = clean_data["current_stock"].mean()
    avg_demand = clean_data["avg_daily_demand"].mean()

    status_counts = (
        clean_data[TARGET]
        .value_counts()
        .to_dict()
    )

    stockout_count = status_counts.get("stockout", 0)
    overstock_count = status_counts.get("overstock", 0)

    stockout_pct = (
        stockout_count / total_records * 100
        if total_records else 0
    )

    overstock_pct = (
        overstock_count / total_records * 100
        if total_records else 0
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        metric_card(
            "Total Records",
            f"{total_records:,}",
            "Inventory observations"
        )

    with col2:
        metric_card(
            "Stockout Records",
            f"{stockout_count:,}",
            f"{stockout_pct:.1f}% of observations"
        )

    with col3:
        metric_card(
            "Overstock Records",
            f"{overstock_count:,}",
            f"{overstock_pct:.1f}% of observations"
        )

    with col4:
        metric_card(
            "Average Daily Demand",
            f"{avg_demand:.2f}",
            "Units per day"
        )

    st.markdown("")

    col5, col6, col7, col8 = st.columns(4)

    with col5:
        metric_card(
            "Total Stock",
            f"{total_stock:,.0f}",
            "Sum of current stock records"
        )

    with col6:
        metric_card(
            "Average Stock",
            f"{avg_stock:.2f}",
            "Units per observation"
        )

    with col7:
        metric_card(
            "Model Accuracy",
            f"{selected_metrics['Accuracy'] * 100:.2f}%",
            selected_model_name
        )

    with col8:
        metric_card(
            "Weighted F1 Score",
            f"{selected_metrics['F1 Score'] * 100:.2f}%",
            "Test-set evaluation"
        )

    st.markdown(
        '<div class="section-title">Inventory Risk Analysis</div>',
        unsafe_allow_html=True
    )

    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.markdown("#### Stock Status Distribution")

        status_df = pd.DataFrame({
            "Stock Status": [
                format_label(label)
                for label in status_counts.keys()
            ],
            "Records": list(status_counts.values())
        })

        if not status_df.empty:
            st.bar_chart(
                status_df.set_index("Stock Status"),
                height=300
            )

    with chart_col2:
        st.markdown("#### Inventory by Category")

        if "item_category" in clean_data.columns:
            category_df = (
                clean_data.groupby("item_category")
                .size()
                .reset_index(name="Records")
                .set_index("item_category")
            )

            st.bar_chart(category_df, height=300)

    chart_col3, chart_col4 = st.columns(2)

    with chart_col3:
        st.markdown("#### Stock vs Reorder Point")

        stock_chart_data = clean_data[
            ["current_stock", "reorder_point"]
        ].head(100).reset_index(drop=True)

        st.line_chart(stock_chart_data, height=300)

    with chart_col4:
        st.markdown("#### Demand Distribution")

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(
            clean_data["avg_daily_demand"].dropna(),
            bins=25,
            edgecolor="white"
        )
        ax.set_xlabel("Average Daily Demand")
        ax.set_ylabel("Number of Records")
        ax.set_title("Distribution of Daily Demand")
        ax.grid(axis="y", alpha=0.2)
        st.pyplot(fig)
        plt.close(fig)

    st.markdown(
        '<div class="section-title">Key Inventory Indicators</div>',
        unsafe_allow_html=True
    )

    indicator_col1, indicator_col2, indicator_col3 = st.columns(3)

    with indicator_col1:
        st.markdown("#### Reorder Alerts")
        below_reorder = (
            clean_data["current_stock"]
            <= clean_data["reorder_point"]
        ).sum()

        st.metric(
            "Records at or below reorder point",
            f"{below_reorder:,}"
        )

    with indicator_col2:
        st.markdown("#### Low Stock Coverage")

        low_coverage = (
            clean_data["current_stock"]
            < clean_data["avg_daily_demand"]
            * clean_data["lead_time_days"]
        ).sum()

        st.metric(
            "Records below lead-time demand",
            f"{low_coverage:,}"
        )

    with indicator_col3:
        st.markdown("#### Supplier Reliability")

        if "supplier_reliability" in clean_data.columns:
            low_supplier = (
                clean_data["supplier_reliability"]
                .astype(str)
                .str.lower()
                .eq("low")
                .sum()
            )

            st.metric(
                "Low reliability records",
                f"{low_supplier:,}"
            )

    st.markdown(
        '<div class="section-title">Recent Inventory Records</div>',
        unsafe_allow_html=True
    )

    display_columns = [
        column for column in [
            "current_stock",
            "avg_daily_demand",
            "lead_time_days",
            "reorder_point",
            "item_category",
            "supplier_reliability",
            TARGET
        ]
        if column in clean_data.columns
    ]

    st.dataframe(
        clean_data[display_columns].tail(15).reset_index(drop=True),
        use_container_width=True,
        hide_index=True
    )

    csv_data = clean_data.to_csv(index=False).encode("utf-8")

    st.download_button(
        "⬇️ Download Inventory Dataset",
        data=csv_data,
        file_name="stock_risk_inventory_data.csv",
        mime="text/csv"
    )

# ============================================================
# 14. PREDICT STOCK RISK
# ============================================================

elif page == "Predict Stock Risk":

    st.markdown(
        '<div class="section-title">Stock Risk Prediction</div>',
        unsafe_allow_html=True
    )

    st.write(
        "Enter the inventory details below to predict whether the "
        "stock is at risk of stockout or overstock."
    )

    st.info(
        f"Selected model: {selected_model_name}. "
        "Complete the inputs and click Predict Stock Risk."
    )

    with st.form("stock_prediction_form"):

        st.markdown("### Inventory Information")

        col1, col2, col3 = st.columns(3)

        with col1:
            current_stock = st.number_input(
                "Current Stock (units)",
                min_value=0.0,
                value=100.0,
                step=1.0
            )

            avg_daily_demand = st.number_input(
                "Average Daily Demand",
                min_value=0.0,
                value=10.0,
                step=0.5
            )

            lead_time_days = st.number_input(
                "Supplier Lead Time (days)",
                min_value=0.0,
                value=7.0,
                step=1.0
            )

        with col2:
            reorder_point = st.number_input(
                "Reorder Point",
                min_value=0.0,
                value=80.0,
                step=1.0
            )

            sales_last_30_days = st.number_input(
                "Sales in Last 30 Days",
                min_value=0.0,
                value=300.0,
                step=1.0
            )

            stock_turnover_ratio = st.number_input(
                "Stock Turnover Ratio",
                min_value=0.0,
                value=2.0,
                step=0.1
            )

        with col3:
            forecast_error = st.number_input(
                "Forecast Error",
                value=0.0,
                step=0.5
            )

            season = st.selectbox(
                "Season",
                ["peak", "off-peak"]
            )

            item_category = st.selectbox(
                "Item Category",
                ["essential", "non-essential"]
            )

            supplier_reliability = st.selectbox(
                "Supplier Reliability",
                ["high", "low"]
            )

        submitted = st.form_submit_button(
            "🔍 Predict Stock Risk",
            use_container_width=True
        )

    if submitted:

        input_data = pd.DataFrame([{
            "current_stock": current_stock,
            "avg_daily_demand": avg_daily_demand,
            "lead_time_days": lead_time_days,
            "reorder_point": reorder_point,
            "sales_last_30_days": sales_last_30_days,
            "stock_turnover_ratio": stock_turnover_ratio,
            "forecast_error": forecast_error,
            "season": season,
            "item_category": item_category,
            "supplier_reliability": supplier_reliability
        }])[FEATURES]

        try:
            predicted_risk = get_prediction(
                selected_model,
                input_data,
                label_encoder
            )

            probabilities = get_probabilities(
                selected_model,
                input_data,
                label_encoder
            )

            st.divider()
            st.markdown("### Prediction Results")

            result_col1, result_col2 = st.columns([1, 2])

            with result_col1:
                if predicted_risk == "stockout":
                    st.markdown(
                        """
                        <div class="risk-high">
                            ⚠️ STOCKOUT RISK DETECTED
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                elif predicted_risk == "overstock":
                    st.markdown(
                        """
                        <div class="risk-low">
                            📦 OVERSTOCK RISK DETECTED
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                else:
                    st.success(
                        f"Predicted status: {format_label(predicted_risk)}"
                    )

                st.metric(
                    "Selected Model",
                    selected_model_name
                )

            with result_col2:
                st.markdown("#### Input Summary")

                summary = pd.DataFrame({
                    "Inventory Metric": [
                        "Current Stock",
                        "Average Daily Demand",
                        "Lead Time",
                        "Reorder Point",
                        "Sales in Last 30 Days",
                        "Stock Turnover Ratio",
                        "Forecast Error",
                        "Season",
                        "Item Category",
                        "Supplier Reliability"
                    ],
                    "Value": [
                        current_stock,
                        avg_daily_demand,
                        lead_time_days,
                        reorder_point,
                        sales_last_30_days,
                        stock_turnover_ratio,
                        forecast_error,
                        season,
                        item_category,
                        supplier_reliability
                    ]
                })

                st.dataframe(
                    summary,
                    use_container_width=True,
                    hide_index=True
                )

            if probabilities:
                st.markdown("#### Predicted Class Probabilities")

                probability_df = pd.DataFrame({
                    "Stock Status": [
                        format_label(key)
                        for key in probabilities.keys()
                    ],
                    "Probability": list(probabilities.values())
                })

                st.bar_chart(
                    probability_df.set_index("Stock Status")
                )

                st.caption(
                    "Probabilities are model estimates, not guarantees "
                    "of future inventory outcomes."
                )

            st.markdown("### Recommended Actions")

            recommendations = make_recommendations(
                predicted_risk,
                current_stock,
                avg_daily_demand,
                lead_time_days,
                reorder_point,
                supplier_reliability
            )

            for recommendation in recommendations:
                st.markdown(f"- {recommendation}")

            result_export = input_data.copy()
            result_export["predicted_stock_status"] = predicted_risk
            result_export["selected_model"] = selected_model_name

            if probabilities:
                for status, probability in probabilities.items():
                    result_export[
                        f"probability_{status}"
                    ] = probability

            st.download_button(
                "⬇️ Download Prediction Result",
                data=result_export.to_csv(index=False).encode("utf-8"),
                file_name="stock_risk_prediction.csv",
                mime="text/csv"
            )

        except Exception as error:
            st.error("Prediction could not be completed.")
            st.exception(error)

# ============================================================
# 15. MODEL PERFORMANCE
# ============================================================

elif page == "Model Performance":

    st.markdown(
        '<div class="section-title">Model Performance & Evaluation</div>',
        unsafe_allow_html=True
    )

    st.write(
        "Compare the available classification models using the same "
        "test split created from the inventory dataset."
    )

    st.info(
        "The saved XGBoost model may have been trained using a different "
        "split or preprocessing workflow. Compare its results with care."
    )

    results = []

    for model_name in MODEL_OPTIONS:
        model = models.get(model_name)

        if model is None:
            continue

        try:
            metrics = get_model_metrics(
                model,
                X_test,
                y_test
            )

            results.append({
                "Model": model_name,
                "Accuracy": metrics["Accuracy"],
                "Precision": metrics["Precision"],
                "Recall": metrics["Recall"],
                "F1 Score": metrics["F1 Score"]
            })

        except Exception:
            continue

    if results:
        results_df = pd.DataFrame(results)

        st.markdown("### Model Comparison")

        formatted_results = results_df.copy()

        for column in [
            "Accuracy",
            "Precision",
            "Recall",
            "F1 Score"
        ]:
            formatted_results[column] = (
                formatted_results[column] * 100
            ).round(2)

        st.dataframe(
            formatted_results,
            use_container_width=True,
            hide_index=True
        )

        st.markdown("### Performance Metrics")

        metric_choice = st.selectbox(
            "Choose a metric",
            ["Accuracy", "Precision", "Recall", "F1 Score"]
        )

        chart_data = results_df[
            ["Model", metric_choice]
        ].set_index("Model")

        st.bar_chart(
            chart_data,
            height=350
        )

        best_model_row = results_df.loc[
            results_df["F1 Score"].idxmax()
        ]

        st.success(
            f"Highest weighted F1 score in this comparison: "
            f"{best_model_row['Model']} "
            f"({best_model_row['F1 Score'] * 100:.2f}%)."
        )

        st.caption(
            "Choose a model based on the business cost of missed "
            "stockouts, false alerts, and unnecessary replenishment. "
            "Accuracy alone may not be sufficient."
        )

        st.markdown("### Selected Model: Confusion Matrix")

        try:
            selected_predictions = selected_metrics["Predictions"]

            cm = confusion_matrix(
                y_test,
                selected_predictions,
                labels=np.arange(len(label_encoder.classes_))
            )

            cm_df = pd.DataFrame(
                cm,
                index=[
                    f"Actual: {format_label(label)}"
                    for label in label_encoder.classes_
                ],
                columns=[
                    f"Predicted: {format_label(label)}"
                    for label in label_encoder.classes_
                ]
            )

            st.dataframe(
                cm_df,
                use_container_width=True
            )

            fig, ax = plt.subplots(figsize=(6, 4))

            image = ax.imshow(cm, interpolation="nearest")
            ax.set_title(f"Confusion Matrix — {selected_model_name}")
            ax.set_xlabel("Predicted Class")
            ax.set_ylabel("Actual Class")

            ax.set_xticks(
                range(len(label_encoder.classes_))
            )
            ax.set_xticklabels(
                [format_label(x) for x in label_encoder.classes_],
                rotation=20
            )

            ax.set_yticks(
                range(len(label_encoder.classes_))
            )
            ax.set_yticklabels(
                [format_label(x) for x in label_encoder.classes_]
            )

            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax.text(
                        j,
                        i,
                        str(cm[i, j]),
                        ha="center",
                        va="center"
                    )

            fig.colorbar(image, ax=ax)
            st.pyplot(fig)
            plt.close(fig)

            st.markdown("### Classification Report")

            report = classification_report(
                y_test,
                selected_predictions,
                labels=np.arange(len(label_encoder.classes_)),
                target_names=[
                    format_label(label)
                    for label in label_encoder.classes_
                ],
                output_dict=True,
                zero_division=0
            )

            report_df = pd.DataFrame(report).transpose()

            st.dataframe(
                report_df.round(3),
                use_container_width=True
            )

        except Exception as error:
            st.warning(
                f"Some evaluation details could not be displayed: {error}"
            )

    else:
        st.warning(
            "No model evaluation results are available."
        )

# ============================================================
# 16. BUSINESS RECOMMENDATIONS
# ============================================================

elif page == "Business Recommendations":

    st.markdown(
        '<div class="section-title">Business Recommendations</div>',
        unsafe_allow_html=True
    )

    st.write(
        "Use inventory indicators and model predictions to support "
        "purchasing, replenishment, and stock-management decisions."
    )

    recommendation_tabs = st.tabs([
        "Stockout Prevention",
        "Overstock Management",
        "Supplier Management",
        "Demand Planning"
    ])

    with recommendation_tabs[0]:
        st.markdown("### Stockout Prevention")

        st.markdown("""
        - Prioritize products predicted to have stockout risk.
        - Compare current inventory against the reorder point.
        - Review demand expected during supplier lead time.
        - Confirm open purchase orders and delivery schedules.
        - Escalate essential items with low inventory coverage.
        """)

    with recommendation_tabs[1]:
        st.markdown("### Overstock Management")

        st.markdown("""
        - Review products with high stock and low recent sales.
        - Avoid placing repeat orders without checking demand.
        - Consider approved promotions for suitable slow-moving items.
        - Review holding costs, expiry dates, and storage capacity.
        - Adjust replenishment plans using updated demand forecasts.
        """)

    with recommendation_tabs[2]:
        st.markdown("### Supplier Management")

        st.markdown("""
        - Track supplier delivery reliability and lead times.
        - Follow up on delayed orders for critical products.
        - Maintain backup suppliers where business policy permits.
        - Compare expected delivery dates with inventory coverage.
        - Review supplier performance using historical records.
        """)

    with recommendation_tabs[3]:
        st.markdown("### Demand Planning")

        st.markdown("""
        - Monitor changes in average daily demand and recent sales.
        - Review seasonal effects on purchasing requirements.
        - Track forecast errors to identify demand-planning issues.
        - Compare predictions with actual inventory outcomes.
        - Retrain and validate models when new data becomes available.
        """)

    st.divider()

    st.markdown("### Live Inventory Checks")

    check_col1, check_col2 = st.columns(2)

    with check_col1:
        reorder_alerts = (
            clean_data["current_stock"]
            <= clean_data["reorder_point"]
        ).sum()

        st.metric(
            "At or Below Reorder Point",
            f"{reorder_alerts:,}"
        )

    with check_col2:
        lead_time_alerts = (
            clean_data["current_stock"]
            < clean_data["avg_daily_demand"]
            * clean_data["lead_time_days"]
        ).sum()

        st.metric(
            "Below Lead-Time Demand",
            f"{lead_time_alerts:,}"
        )

    st.caption(
        "These counts are calculated from the dataset. They are "
        "inventory indicators, not independent proof of future losses."
    )

# ============================================================
# 17. FOOTER
# ============================================================

st.divider()

st.markdown(
    """
    <div style="text-align:center; color:#78849A; font-size:12px;">
        Stock Risk Intelligence | Machine Learning Inventory Analytics
        <br>
        Predictions are decision-support estimates. Validate results
        against real stock levels and operational requirements.
    </div>
    """,
    unsafe_allow_html=True
)
