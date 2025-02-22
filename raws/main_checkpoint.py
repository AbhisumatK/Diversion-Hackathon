import streamlit as st
import pandas as pd
import numpy as np
import optuna
import xgboost as xgb
import lightgbm as lgb
from concurrent.futures import ThreadPoolExecutor
from joblib import Memory
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, VotingClassifier,
    RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
)
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, r2_score
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
import warnings

warnings.filterwarnings("ignore")

st.set_page_config(layout="wide")
st.title("Optimized Model Trainer with XGBoost, LightGBM & Optuna")

# Initialize session state variables
for key in ["df", "best_model", "transformer", "chosen_features", "is_classification", 
            "best_model_name", "test_score", "poly_degree", "dataset_type", "target_column", "prediction_result"]:
    if key not in st.session_state:
        st.session_state[key] = None

memory = Memory(location="cache_dir", verbose=0)

def encode_features(df):
    for col in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
    return df

def automatic_feature_selection(X, y, problem_type):
    if problem_type == "Classification":
        scores = mutual_info_classif(X, y)
    else:
        scores = mutual_info_regression(X, y)
    feature_scores = pd.Series(scores, index=X.columns).sort_values(ascending=False)
    return feature_scores[feature_scores > 0.01].index.tolist()

with st.sidebar:
    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])
    dataset_type = st.radio("Dataset Type", ["Numeric"])
    st.session_state.dataset_type = dataset_type
    problem_type_option = st.radio("Problem Type", ["Auto Detect", "Classification", "Regression"])
    model_choice = st.selectbox("Select Model", [
        "Auto", "Random Forest", "Gradient Boosting", "XGBoost", "LightGBM", "SVM", "KNN", "Linear Regression"
    ])
    apply_poly = st.checkbox("Apply Polynomial Features", value=False)
    if apply_poly:
        degree = st.slider("Polynomial Degree", min_value=2, max_value=3, value=2, step=1)
        st.session_state.poly_degree = degree

    if uploaded_file:
        df = pd.read_csv(uploaded_file).dropna()
        df = encode_features(df)
        st.session_state.df = df
        st.write("### Cleaned Dataset Preview", df.head())
        target_column = st.selectbox("Select Target Column", df.columns)
        st.session_state.target_column = target_column
        candidate_features = [col for col in df.columns if col != target_column]

        y = df[target_column]
        problem_type = "Classification" if y.nunique() <= 10 else "Regression"
        st.session_state.is_classification = (problem_type == "Classification")

        # Automatic feature selection using mutual information
        chosen_features = automatic_feature_selection(df[candidate_features], y, problem_type)
        st.session_state.chosen_features = chosen_features
        st.write("### Automatically Selected Features", chosen_features)

if uploaded_file is None:
    st.session_state.clear()
    st.write("No dataset uploaded. Please upload a CSV file.")
    st.stop()

df = st.session_state.df
target_column = st.session_state.target_column
X = df[st.session_state.chosen_features]
y = df[target_column]

# Create a transformation pipeline: PolynomialFeatures (if enabled) followed by StandardScaler.
if apply_poly:
    transformer = make_pipeline(
        PolynomialFeatures(degree=st.session_state.poly_degree, include_bias=False),
        StandardScaler()
    )
else:
    transformer = StandardScaler()
X_transformed = transformer.fit_transform(X)
st.session_state.transformer = transformer

X_train, X_test, y_train, y_test = train_test_split(X_transformed, y, test_size=0.2, random_state=42)

def objective(trial):
    param = {
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.2),
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'subsample': trial.suggest_uniform('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.6, 1.0)
    }
    model = xgb.XGBClassifier(**param) if y.nunique() <= 10 else xgb.XGBRegressor(**param)
    model.fit(X_train, y_train)
    score = accuracy_score(y_test, model.predict(X_test)) if y.nunique() <= 10 else r2_score(y_test, model.predict(X_test))
    return score

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=10)
best_params = study.best_params

if y.nunique() <= 10:
    models = {
        "Random Forest": (RandomForestClassifier(), {"n_estimators": [50, 100, 200]}),
        "Gradient Boosting": (GradientBoostingClassifier(), {"learning_rate": [0.01, 0.1, 0.2]}),
        "XGBoost": (xgb.XGBClassifier(**best_params), {}),
        "LightGBM": (lgb.LGBMClassifier(), {})
    }
else:
    models = {
        "Random Forest": (RandomForestRegressor(), {"n_estimators": [50, 100, 200]}),
        "Gradient Boosting": (GradientBoostingRegressor(), {"learning_rate": [0.01, 0.1, 0.2]}),
        "XGBoost": (xgb.XGBRegressor(**best_params), {}),
        "LightGBM": (lgb.LGBMRegressor(), {})
    }

def train_model(item):
    name, (model, param_grid) = item
    scoring = "accuracy" if y.nunique() <= 10 else "r2"
    grid = RandomizedSearchCV(model, param_grid, cv=5, scoring=scoring, n_jobs=-1)
    grid.fit(X_train, y_train)
    return name, grid.best_estimator_, grid.best_score_

st.write("### Training Models in Parallel")
if st.button("Train & Evaluate"):
    with ThreadPoolExecutor() as executor:
        results = list(executor.map(train_model, models.items()))
    
    best_model_name, best_model, best_score = max(results, key=lambda x: x[2])
    st.session_state.best_model = best_model
    st.session_state.best_model_name = best_model_name
    st.write(f"**Best Model:** {best_model_name} with Score: {best_score:.2f}")

    y_pred = best_model.predict(X_test)
    metric = accuracy_score if y.nunique() <= 10 else r2_score
    st.session_state.test_score = metric(y_test, y_pred) * 100
    st.write(f"### {best_model_name} Performance: {st.session_state.test_score:.2f}%")

# Prediction Section
st.write("## Predict with Best Model")
if st.session_state.best_model is None:
    st.write("Train a model first to make predictions.")
else:
    st.write("Select the features for which you want to provide input. For the rest, default (mean) values will be used.")
    # Let the user select which features to provide manually.
    selected_features = st.multiselect("Select features for manual input", st.session_state.chosen_features)
    prediction_inputs = {}
    for feature in selected_features:
        default_val = st.session_state.df[feature].mean()  # default value from training data
        # Unique key to persist input value
        prediction_inputs[feature] = st.number_input(f"Enter {feature}", value=default_val, key=f"input_{feature}")
    
    if st.button("Make Prediction"):
        # Build a full input vector in the same order as the chosen features
        full_input = []
        for feature in st.session_state.chosen_features:
            if feature in prediction_inputs:
                full_input.append(prediction_inputs[feature])
            else:
                # Use the default value for features not manually entered
                full_input.append(st.session_state.df[feature].mean())
        input_array = np.array(full_input).reshape(1, -1)
        # Transform the input using the same pipeline as during training
        input_array_transformed = st.session_state.transformer.transform(input_array)
        prediction = st.session_state.best_model.predict(input_array_transformed)
        st.write(f"### Prediction Result: {prediction[0]}")