import streamlit as st
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from joblib import Memory
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.metrics import r2_score
import lightgbm as lgb

st.set_page_config(layout="wide")
st.title("Optimized Machine Learning Model Trainer")

# Initialize session state
for key in ["df", "best_model", "scaler", "chosen_features", "best_model_name", "test_score", "poly", "poly_degree", "target_column"]:
    if key not in st.session_state:
        st.session_state[key] = None

memory = Memory(location="cache_dir", verbose=0)

@memory.cache
def encode_series(s):
    if s.dtype == 'O':
        le = LabelEncoder()
        return pd.Series(le.fit_transform(s))
    return s

with st.sidebar:
    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])
    apply_poly = st.checkbox("Apply Polynomial Features", value=False)
    if apply_poly:
        degree = st.slider("Polynomial Degree", min_value=2, max_value=3, value=2, step=1)
        st.session_state.poly_degree = degree
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file).dropna()
        st.session_state.df = df
        st.write("### Available Columns:", df.columns.tolist())
        target_column = st.selectbox("Select Target Column", df.columns)
        st.session_state.target_column = target_column
        chosen_features = st.multiselect("Select Features", [col for col in df.columns if col != target_column])
        st.session_state.chosen_features = chosen_features if chosen_features else df.drop(columns=[target_column]).columns.tolist()

if uploaded_file is None:
    st.session_state.clear()
    st.write("No dataset uploaded. Please upload a CSV file.")
    st.stop()

df = st.session_state.df
target_column = st.session_state.target_column
X = df[st.session_state.chosen_features]
y = df[target_column]

for col in X.columns:
    if X[col].dtype == 'O':
        X[col] = encode_series(X[col])

scaler = StandardScaler()
if apply_poly:
    poly = PolynomialFeatures(degree=st.session_state.poly_degree, include_bias=False)
    X = poly.fit_transform(X)
    st.session_state.poly = poly

X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
st.session_state.scaler = scaler

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def train_model(name, model, param_grid):
    grid = RandomizedSearchCV(model, param_grid, cv=5, scoring="r2", n_jobs=-1)
    grid.fit(X_train, y_train)
    return name, grid.best_estimator_, grid.best_score_

models = {
    "Random Forest": (RandomForestRegressor(), {"n_estimators": [50, 100, 200]}),
    "Gradient Boosting": (GradientBoostingRegressor(), {"learning_rate": [0.01, 0.1, 0.2]}),
    "LightGBM": (lgb.LGBMRegressor(), {"num_leaves": [31, 50], "learning_rate": [0.01, 0.1, 0.2]})
}

st.write("### Training Models in Parallel")
if st.button("Train & Evaluate"):
    with ThreadPoolExecutor() as executor:
        results = list(executor.map(lambda item: train_model(*item), models.items()))

    best_model_name, best_model, best_score = max(results, key=lambda x: x[2])
    st.session_state.best_model = best_model
    st.session_state.best_model_name = best_model_name
    st.write(f"**Best Model:** {best_model_name} with Score: {best_score:.2f}")

    y_pred = best_model.predict(X_test)
    st.session_state.test_score = r2_score(y_test, y_pred) * 100
    st.write(f"### {best_model_name} Performance: {st.session_state.test_score:.2f}%")

if st.session_state.best_model:
    st.write(f"### Model in use: {st.session_state.best_model_name} (Score: {st.session_state.test_score:.2f}%)")

st.write("### Make Predictions")
input_values = []
for col in st.session_state.chosen_features:
    input_values.append(st.number_input(f"{col}", value=float(df[col].median())))

if st.button("Predict"):
    input_df = pd.DataFrame([input_values], columns=X.columns)
    prediction = st.session_state.best_model.predict(input_df)[0]
    st.write(f"### Predicted Value: {prediction:.2f}")
