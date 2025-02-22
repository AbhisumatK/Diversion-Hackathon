import streamlit as st
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from joblib import Memory
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier, VotingClassifier,
                              RandomForestRegressor, GradientBoostingRegressor, VotingRegressor, StackingRegressor, StackingClassifier)
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.feature_selection import RFE
from sklearn.metrics import accuracy_score, r2_score

st.set_page_config(layout="wide")
st.title("Optimized Ensemble Model Trainer for Numerical Data with Multi Processing")

# Initialize session state variables
for key in ["df", "best_model", "scaler", "chosen_features", "is_classification",
            "best_model_name", "test_score", "poly", "poly_degree", "target_column",
            "prediction_result", "cleaned_df"]:
    if key not in st.session_state:
        st.session_state[key] = None

memory = Memory(location="cache_dir", verbose=0)

# Sidebar: File Upload & Options
with st.sidebar:
    st.header("Dataset & Options")
    uploaded_file = st.file_uploader("Upload CSV File (Numerical Data)", type=["csv"])
    apply_poly = st.checkbox("Apply Polynomial Features", value=False)
    if apply_poly:
        degree = st.slider("Polynomial Degree", min_value=2, max_value=3, value=2, step=1)
        st.session_state.poly_degree = degree

    if uploaded_file:
        # Read data
        df = pd.read_csv(uploaded_file)
        
        # Data cleaning:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        valid_cols = [col for col in numeric_cols if df[col].nunique() > 1]
        cleaned_df = df[valid_cols].dropna()
        st.session_state.cleaned_df = cleaned_df

        st.write("### Cleaned Dataset (Valid Numeric Columns)")
        st.dataframe(cleaned_df)

        target_column = st.selectbox("Select Target Column", cleaned_df.columns.tolist())
        st.session_state.target_column = target_column
        candidate_features = [col for col in cleaned_df.columns if col != target_column]
        corr = cleaned_df[candidate_features].corrwith(cleaned_df[target_column]).abs()
        relevant_features = corr[corr > 0.1].index.tolist()
        st.session_state.chosen_features = relevant_features
        st.write("Automatically selected features based on correlation threshold:", relevant_features)

if uploaded_file is None:
    st.write("Please upload a CSV file with numerical data.")
    st.stop()

# Data Preparation
df = st.session_state.cleaned_df
target_column = st.session_state.target_column
X = df[st.session_state.chosen_features]
y = df[target_column]

if X.empty or y.empty:
    st.write("Insufficient data to train the model. Please ensure there are valid features and target.")
    st.stop()

is_classification = y.nunique() <= 10
st.session_state.is_classification = is_classification

scaler = StandardScaler()
if apply_poly:
    poly = PolynomialFeatures(degree=st.session_state.poly_degree, include_bias=False)
    X_transformed = poly.fit_transform(X)
    st.session_state.poly = poly
else:
    X_transformed = X.values
X_scaled = scaler.fit_transform(X_transformed)
st.session_state.scaler = scaler

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Feature Selection
selector = RFE(LinearRegression() if not is_classification else LogisticRegression(), n_features_to_select=5)
X_train = selector.fit_transform(X_train, y_train)
X_test = selector.transform(X_test)

# Model Training
def train_model(name, model, param_grid):
    grid = RandomizedSearchCV(model, param_grid, cv=5,
                              scoring="accuracy" if is_classification else "r2",
                              n_jobs=-1)
    grid.fit(X_train, y_train)
    return name, grid.best_estimator_, grid.best_score_

candidate_models = {
    "Random Forest": (RandomForestRegressor() if not is_classification else RandomForestClassifier(), 
                       {"n_estimators": [50, 100, 200], "max_depth": [None, 5, 10]}),
    "Gradient Boosting": (GradientBoostingRegressor() if not is_classification else GradientBoostingClassifier(), 
                           {"n_estimators": [50, 100, 200], "learning_rate": [0.01, 0.1, 0.2]}),
    "KNN": (KNeighborsRegressor() if not is_classification else KNeighborsClassifier(), 
             {"n_neighbors": [3, 5, 7]})
}

st.write("### Training Candidate Models in Parallel")
with ThreadPoolExecutor() as executor:
    results = list(executor.map(lambda item: train_model(item[0], item[1][0], item[1][1]),
                                candidate_models.items()))

estimators = [(name, model) for name, model, score in results]
ensemble_model = StackingRegressor(estimators=estimators, final_estimator=LinearRegression()) if not is_classification else StackingClassifier(estimators=estimators, final_estimator=LogisticRegression())

ensemble_model.fit(X_train, y_train)
st.session_state.best_model = ensemble_model
st.session_state.best_model_name = "Stacking Ensemble"

y_pred = ensemble_model.predict(X_test)
test_score = r2_score(y_test, y_pred) * 100 if not is_classification else accuracy_score(y_test, y_pred) * 100
st.session_state.test_score = test_score
st.write(f"### Ensemble Model Performance: {test_score:.2f}%")

st.write("## Make a Prediction")
input_data = {feature: st.number_input(f"Enter {feature}", value=float(df[feature].median())) for feature in st.session_state.chosen_features}

if st.button("Predict"):
    new_data = pd.DataFrame([input_data])
    if apply_poly:
        new_data = st.session_state.poly.transform(new_data)
    new_data = st.session_state.scaler.transform(new_data)
    prediction = st.session_state.best_model.predict(new_data)
    st.session_state.prediction_result = prediction[0]
    st.write(f"### Prediction: {prediction[0]}")
