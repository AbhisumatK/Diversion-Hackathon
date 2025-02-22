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
from sklearn.pipeline import Pipeline
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, VotingClassifier,
    RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
)
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, r2_score
import warnings

warnings.filterwarnings("ignore")

st.set_page_config(layout="wide")
st.title("Optimized Model Trainer with XGBoost, LightGBM & Optuna")

for key in ["df", "best_model", "scaler", "chosen_features", "is_classification", 
            "best_model_name", "test_score", "poly", "poly_degree", "dataset_type", "target_column", "prediction_result"]:
    if key not in st.session_state:
        st.session_state[key] = None

memory = Memory(location="cache_dir", verbose=0)

def encode_features(df):
    for col in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
    return df

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
        chosen_features = st.multiselect("Select Features", candidate_features)
        
        if not chosen_features:
            encoded_target = df[target_column]
            corr = df[candidate_features].apply(lambda col: col.corr(encoded_target)).abs()
            st.session_state.chosen_features = corr[corr > 0.1].index.tolist()
        else:
            st.session_state.chosen_features = chosen_features

if uploaded_file is None:
    st.session_state.clear()
    st.write("No dataset uploaded. Please upload a CSV file.")
    st.stop()

df = st.session_state.df
target_column = st.session_state.target_column
X = df[st.session_state.chosen_features]
y = df[target_column]

scaler = StandardScaler()
if apply_poly:
    poly = PolynomialFeatures(degree=st.session_state.poly_degree, include_bias=False)
    X = poly.fit_transform(X)
    st.session_state.poly = poly

X = scaler.fit_transform(X)
st.session_state.scaler = scaler

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

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

models = {
    "Random Forest": (RandomForestClassifier() if y.nunique() <= 10 else RandomForestRegressor(), {"n_estimators": [50, 100, 200]}),
    "Gradient Boosting": (GradientBoostingClassifier() if y.nunique() <= 10 else GradientBoostingRegressor(), {"learning_rate": [0.01, 0.1, 0.2]}),
    "XGBoost": (xgb.XGBClassifier(**best_params) if y.nunique() <= 10 else xgb.XGBRegressor(**best_params), {}),
    "LightGBM": (lgb.LGBMClassifier() if y.nunique() <= 10 else lgb.LGBMRegressor(), {})
}

def train_model(name, model, param_grid):
    grid = RandomizedSearchCV(model, param_grid, cv=5, scoring="accuracy" if y.nunique() <= 10 else "r2", n_jobs=-1)
    grid.fit(X_train, y_train)
    return name, grid.best_estimator_, grid.best_score_

st.write("### Training Models in Parallel")
if st.button("Train & Evaluate"):
    with ThreadPoolExecutor() as executor:
        results = list(executor.map(lambda item: train_model(*item), models.items()))
    
    best_model_name, best_model, best_score = max(results, key=lambda x: x[2])
    st.session_state.best_model = best_model
    st.session_state.best_model_name = best_model_name
    st.write(f"**Best Model:** {best_model_name} with Score: {best_score:.2f}")

    y_pred = best_model.predict(X_test)
    metric = accuracy_score if y.nunique() <= 10 else r2_score
    st.session_state.test_score = metric(y_test, y_pred) * 100
    st.write(f"### {best_model_name} Performance: {st.session_state.test_score:.2f}%")

st.write("## Predict with Best Model")
input_data = [st.number_input(f"Enter {feature}", value=0.0) for feature in st.session_state.chosen_features]
if st.button("Make Prediction"):
    input_array = np.array(input_data).reshape(1, -1)
    input_array = scaler.transform(input_array)
    if apply_poly:
        input_array = poly.transform(input_array)
    prediction = st.session_state.best_model.predict(input_array)
    st.write(f"### Prediction Result: {prediction[0]}")