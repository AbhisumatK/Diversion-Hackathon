import streamlit as st
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from joblib import Memory
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier, VotingClassifier,
                              RandomForestRegressor, GradientBoostingRegressor, VotingRegressor)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, r2_score

st.set_page_config(layout="wide")
st.title("Optimized Model Trainer with Predictions")

# Initialize session state
for key in ["df", "best_model", "scaler", "chosen_features", "is_classification", 
            "best_model_name", "test_score", "poly", "poly_degree", "dataset_type", "target_column", "prediction_result"]:
    if key not in st.session_state:
        st.session_state[key] = None

memory = Memory(location="cache_dir", verbose=0)

@memory.cache
def encode_series(s):
    if s.dtype == 'O':
        le = LabelEncoder()
        return pd.Series(le.fit_transform(s))
    return s

# Sidebar UI
with st.sidebar:
    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])
    dataset_type = st.radio("Dataset Type", ["Numeric", "Textual"])
    st.session_state.dataset_type = dataset_type
    problem_type_option = st.radio("Problem Type", ["Auto Detect", "Classification", "Regression"])
    model_choice = st.selectbox("Select Model", [
        "Auto", "Logistic Regression", "Random Forest", "Gradient Boosting", "SVM", "KNN", "Linear Regression"
    ])
    
    apply_poly = st.checkbox("Apply Polynomial Features", value=False) if dataset_type == "Numeric" else False
    if apply_poly:
        degree = st.slider("Polynomial Degree", min_value=2, max_value=3, value=2, step=1)
        st.session_state.poly_degree = degree

    if uploaded_file:
        df = pd.read_csv(uploaded_file).dropna()
        st.session_state.df = df
        st.write("### Available Columns:", df.columns.tolist())
        target_column = st.selectbox("Select Target Column", df.columns)
        st.session_state.target_column = target_column
        candidate_features = [col for col in df.columns if col != target_column]
        chosen_features = st.multiselect("Select Features", candidate_features)
        
        if not chosen_features and dataset_type == "Numeric":
            encoded_target = encode_series(df[target_column])
            corr = df[candidate_features].apply(lambda col: encode_series(col).corr(encoded_target)).abs()
            st.session_state.chosen_features = corr[corr > 0.1].index.tolist()
        else:
            st.session_state.chosen_features = chosen_features

if uploaded_file is None:
    st.session_state.clear()
    st.write("No dataset uploaded. Please upload a CSV file.")
    st.stop()

# Data Processing
df = st.session_state.df
target_column = st.session_state.target_column
X = df[st.session_state.chosen_features]
y = df[target_column]

if dataset_type == "Textual":
    X = X.astype(str).agg(" ".join, axis=1)
else:
    for col in X.columns:
        if X[col].dtype == 'O':
            X[col] = encode_series(X[col])

is_classification = problem_type_option == "Classification" or (problem_type_option == "Auto Detect" and y.nunique() <= 10)
st.session_state.is_classification = is_classification

# Scaling and Polynomial Features
scaler = StandardScaler()
if apply_poly:
    poly = PolynomialFeatures(degree=st.session_state.poly_degree, include_bias=False)
    X = poly.fit_transform(X)
    st.session_state.poly = poly

X = scaler.fit_transform(X) if dataset_type == "Numeric" else X
st.session_state.scaler = scaler

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Selection
def train_model(name, model, param_grid):
    grid = RandomizedSearchCV(model, param_grid, cv=5, scoring="accuracy" if is_classification else "r2", n_jobs=-1)
    grid.fit(X_train, y_train)
    return name, grid.best_estimator_, grid.best_score_

models = {
    "Logistic Regression": (LogisticRegression(), {"C": [0.01, 0.1, 1, 10]}),
    "Random Forest": (RandomForestClassifier() if is_classification else RandomForestRegressor(), {"n_estimators": [50, 100, 200]}),
    "Gradient Boosting": (GradientBoostingClassifier() if is_classification else GradientBoostingRegressor(), {"learning_rate": [0.01, 0.1, 0.2]}),
    "SVM": (SVC(probability=True), {"C": [0.1, 1, 10]}),
    "KNN": (KNeighborsClassifier(), {"n_neighbors": [3, 5, 7]}),
    "Linear Regression": (LinearRegression(), {})
}

st.write("### Training Models in Parallel")
if st.button("Train & Evaluate"):
    with ThreadPoolExecutor() as executor:
        results = list(executor.map(lambda item: train_model(item[0], item[1][0], item[1][1]), models.items()))

    best_model_name, best_model, best_score = max(results, key=lambda x: x[2])
    st.session_state.best_model = best_model
    st.session_state.best_model_name = best_model_name
    st.write(f"**Best Model:** {best_model_name} with Score: {best_score:.2f}")

    y_pred = best_model.predict(X_test)
    metric = accuracy_score if is_classification else r2_score
    st.session_state.test_score = metric(y_test, y_pred) * 100
    st.write(f"### {best_model_name} Performance: {st.session_state.test_score:.2f}%")

if st.session_state.best_model:
    st.write(f"### Model in use: {st.session_state.best_model_name} (Score: {st.session_state.test_score:.2f}%)")

    # Prediction Section
    st.write("## Make Predictions")
    input_data = {}
    for feature in st.session_state.chosen_features:
        input_data[feature] = st.number_input(f"Enter {feature}", value=0.0)

    if st.button("Predict"):
        new_data = pd.DataFrame([input_data])

        # Encode categorical columns
        for col in new_data.columns:
            if new_data[col].dtype == 'O':
                new_data[col] = encode_series(new_data[col])

        # Apply polynomial features if enabled
        if apply_poly:
            new_data = st.session_state.poly.transform(new_data)

        # Scale data
        new_data = st.session_state.scaler.transform(new_data)

        # Predict using the trained model
        prediction = st.session_state.best_model.predict(new_data)

        # Display Prediction
        st.session_state.prediction_result = prediction[0]
        st.write(f"### Prediction: {prediction[0]}")