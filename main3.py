import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
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
st.title("Advanced Prediction Model Trainer with Ensembling & Feature Engineering")

# Initialize session state variables
for key in ["df", "best_model", "scaler", "chosen_features", "is_classification", 
            "best_model_name", "test_score", "poly", "poly_degree", "dataset_type", "target_column", "prediction_result"]:
    if key not in st.session_state:
        st.session_state[key] = None

# Sidebar options
with st.sidebar:
    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])
    dataset_type = st.radio("Dataset Type", ["Numeric", "Textual"])
    st.session_state.dataset_type = dataset_type
    problem_type_option = st.radio("Problem Type", ["Auto Detect", "Classification", "Regression"])
    model_choice = st.selectbox("Select Model", [
        "Auto", "Logistic Regression", "Random Forest", "Gradient Boosting", "SVM", "KNN", "Linear Regression"
    ])
    if dataset_type == "Numeric":
        apply_poly = st.checkbox("Apply Polynomial Feature Engineering", value=False)
        if apply_poly:
            degree = st.number_input("Polynomial Degree", min_value=2, max_value=3, value=2, step=1)
            st.session_state.poly_degree = degree
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        df.dropna(inplace=True)
        st.session_state.df = df
        st.write("### Available Columns:")
        st.write(df.columns.tolist())
        target_column = st.selectbox("Select Target Column", df.columns)
        st.session_state.target_column = target_column
        candidate_features = [col for col in df.columns if col != target_column]
        chosen_features = st.multiselect("Select Features for Training (optional; leave empty to auto-select relevant ones)", candidate_features)
        if not chosen_features and dataset_type == "Numeric":
            def encode_series(s):
                if s.dtype == 'O':
                    le = LabelEncoder()
                    return pd.Series(le.fit_transform(s))
                else:
                    return s
            encoded_target = encode_series(df[target_column])
            corr = df[candidate_features].apply(lambda col: encode_series(col).corr(encoded_target)).abs()
            relevant_features = corr[corr > 0.1].index.tolist()
            st.session_state.chosen_features = relevant_features
            st.write("Automatically selected relevant features:", relevant_features)
        else:
            st.session_state.chosen_features = chosen_features

if uploaded_file is None:
    st.session_state.clear()
    st.write("No dataset uploaded. Please upload a CSV file.")
    st.stop()

# Main area
df = st.session_state.df
st.write("### Cleaned Dataset")
st.dataframe(df)

# --- Data Preparation ---
if not st.session_state.chosen_features:
    st.error("No features selected. Please select at least one feature (excluding the target).")
else:
    target_column = st.session_state.target_column
    if st.session_state.dataset_type == "Textual":
        X_text = df[st.session_state.chosen_features].astype(str).agg(" ".join, axis=1)
        X_processed = X_text
        y = df[target_column]
    else:
        X_df = df[st.session_state.chosen_features].copy()
        y = df[target_column]
        for col in X_df.columns:
            if X_df[col].dtype == 'O':
                le = LabelEncoder()
                X_df[col] = le.fit_transform(X_df[col])
        X_processed = X_df

    auto_classification = True
    if (df[target_column].dtype in [np.int64, np.float64]) and (df[target_column].nunique() > 10):
        auto_classification = False

    if problem_type_option == "Classification":
        is_classification = True
    elif problem_type_option == "Regression":
        is_classification = False
    else:
        is_classification = auto_classification
    st.session_state.is_classification = is_classification

    if is_classification and y.dtype == 'O':
        le = LabelEncoder()
        y = le.fit_transform(y)

    if st.session_state.dataset_type == "Numeric":
        scaler = StandardScaler()
        if st.session_state.poly_degree is not None:
            poly = PolynomialFeatures(degree=st.session_state.poly_degree, include_bias=False)
            X_poly = poly.fit_transform(X_processed)
            st.session_state.poly = poly
            X_final = scaler.fit_transform(X_poly)
        else:
            X_final = scaler.fit_transform(X_processed)
        st.session_state.scaler = scaler
    else:
        tfidf = TfidfVectorizer()
        X_final = tfidf.fit_transform(X_processed)
        st.session_state.scaler = tfidf

    X_train, X_test, y_train, y_test = train_test_split(X_final, y, test_size=0.2, random_state=42)
    st.write("### Preprocessed Data Shape:", X_final.shape)

    st.write("### Train & Evaluate Model")
    if st.button("Train & Evaluate"):
        # --- Model Training & Ensembling ---
        if st.session_state.dataset_type == "Textual":
            from sklearn.pipeline import Pipeline
            if is_classification:
                from sklearn.ensemble import VotingClassifier
                candidate_models = {
                    "Logistic Regression": Pipeline([("tfidf", TfidfVectorizer()), ("clf", LogisticRegression())]),
                    "Random Forest": Pipeline([("tfidf", TfidfVectorizer()), ("clf", RandomForestClassifier())]),
                    "Gradient Boosting": Pipeline([("tfidf", TfidfVectorizer()), ("clf", GradientBoostingClassifier())]),
                    "SVM": Pipeline([("tfidf", TfidfVectorizer()), ("clf", SVC(probability=True))]),
                    "KNN": Pipeline([("tfidf", TfidfVectorizer()), ("clf", KNeighborsClassifier())])
                }
                param_grids = {
                    "Logistic Regression": {"clf__C": [0.01, 0.1, 1, 10]},
                    "Random Forest": {"clf__n_estimators": [50, 100], "clf__max_depth": [None, 5, 10]},
                    "Gradient Boosting": {"clf__n_estimators": [50, 100], "clf__learning_rate": [0.01, 0.1, 1]},
                    "SVM": {"clf__C": [0.1, 1, 10], "clf__kernel": ["linear", "rbf"]},
                    "KNN": {"clf__n_neighbors": [3, 5, 7]}
                }
                scoring = "accuracy"
                min_class_count = np.min(np.bincount(y_train))
                cv_splits = min(5, min_class_count) if min_class_count >= 2 else 2
                tuned_estimators = []
                best_cv_score = -np.inf
                for name, pipe in candidate_models.items():
                    grid = GridSearchCV(pipe, param_grids[name], cv=cv_splits, scoring=scoring, n_jobs=-1)
                    grid.fit(X_train, y_train)
                    tuned_estimators.append((name, grid.best_estimator_))
                    st.write(f"**{name} best CV Score:** {grid.best_score_*100:.2f}% with params {grid.best_params_}")
                    if grid.best_score_ > best_cv_score:
                        best_cv_score = grid.best_score_
                ensemble_model = VotingClassifier(estimators=tuned_estimators, voting="soft")
                best_model = ensemble_model
                best_model_name = "VotingClassifier (Text Ensemble)"
                st.session_state.best_model_name = best_model_name
            else:
                from sklearn.ensemble import VotingRegressor
                candidate_models = {
                    "Linear Regression": Pipeline([("tfidf", TfidfVectorizer()), ("reg", LinearRegression())]),
                    "Random Forest": Pipeline([("tfidf", TfidfVectorizer()), ("reg", RandomForestRegressor())]),
                    "Gradient Boosting": Pipeline([("tfidf", TfidfVectorizer()), ("reg", GradientBoostingRegressor())])
                }
                param_grids = {
                    "Linear Regression": {},
                    "Random Forest": {"reg__n_estimators": [50, 100], "reg__max_depth": [None, 5, 10]},
                    "Gradient Boosting": {"reg__n_estimators": [50, 100], "reg__learning_rate": [0.01, 0.1, 1]}
                }
                scoring = "r2"
                cv_splits = 5
                tuned_estimators = []
                best_cv_score = -np.inf
                for name, pipe in candidate_models.items():
                    grid = GridSearchCV(pipe, param_grids[name], cv=cv_splits, scoring=scoring, n_jobs=-1)
                    grid.fit(X_train, y_train)
                    tuned_estimators.append((name, grid.best_estimator_))
                    st.write(f"**{name} best CV Score:** {grid.best_score_*100:.2f}% with params {grid.best_params_ if param_grids[name] else '{}'}")
                    if grid.best_score_ > best_cv_score:
                        best_cv_score = grid.best_score_
                ensemble_model = VotingRegressor(estimators=tuned_estimators)
                best_model = ensemble_model
                best_model_name = "VotingRegressor (Text Ensemble)"
                st.session_state.best_model_name = best_model_name
        else:
            if model_choice == "Auto":
                if is_classification:
                    from sklearn.ensemble import VotingClassifier
                    candidate_models = {
                        "Logistic Regression": LogisticRegression(),
                        "Random Forest": RandomForestClassifier(),
                        "Gradient Boosting": GradientBoostingClassifier(),
                        "SVM": SVC(probability=True),
                        "KNN": KNeighborsClassifier()
                    }
                    param_grids = {
                        "Logistic Regression": {"C": [0.01, 0.1, 1, 10]},
                        "Random Forest": {"n_estimators": [50, 100, 200], "max_depth": [None, 5, 10, 20]},
                        "Gradient Boosting": {"n_estimators": [50, 100, 200], "learning_rate": [0.01, 0.1, 0.2], "max_depth": [3, 5, 7]},
                        "SVM": {"C": [0.1, 1, 10], "kernel": ["linear", "rbf"]},
                        "KNN": {"n_neighbors": [3, 5, 7, 9]}
                    }
                    scoring = "accuracy"
                    min_class_count = np.min(np.bincount(y_train))
                    cv_splits = min(5, min_class_count) if min_class_count >= 2 else 2
                    tuned_estimators = []
                    best_cv_score = -np.inf
                    for name, model in candidate_models.items():
                        grid = GridSearchCV(model, param_grids[name], cv=cv_splits, scoring=scoring, n_jobs=-1)
                        grid.fit(X_train, y_train)
                        tuned_estimators.append((name, grid.best_estimator_))
                        st.write(f"**{name} best CV Score:** {grid.best_score_*100:.2f}% with params {grid.best_params_}")
                        if grid.best_score_ > best_cv_score:
                            best_cv_score = grid.best_score_
                    ensemble_model = VotingClassifier(estimators=tuned_estimators, voting="soft")
                    best_model = ensemble_model
                    best_model_name = "VotingClassifier (Ensemble)"
                    st.session_state.best_model_name = best_model_name
                else:
                    from sklearn.ensemble import VotingRegressor
                    candidate_models = {
                        "Linear Regression": LinearRegression(),
                        "Random Forest": RandomForestRegressor(),
                        "Gradient Boosting": GradientBoostingRegressor()
                    }
                    param_grids = {
                        "Linear Regression": {},
                        "Random Forest": {"n_estimators": [50, 100, 200], "max_depth": [None, 5, 10, 20]},
                        "Gradient Boosting": {"n_estimators": [50, 100, 200], "learning_rate": [0.01, 0.1, 0.2], "max_depth": [3, 5, 7]}
                    }
                    scoring = "r2"
                    cv_splits = 5
                    tuned_estimators = []
                    best_cv_score = -np.inf
                    for name, model in candidate_models.items():
                        grid = GridSearchCV(model, param_grids[name], cv=cv_splits, scoring=scoring, n_jobs=-1)
                        grid.fit(X_train, y_train)
                        tuned_estimators.append((name, grid.best_estimator_))
                        st.write(f"**{name} best CV Score:** {grid.best_score_*100:.2f}% with params {grid.best_params_ if param_grids[name] else '{}'}")
                        if grid.best_score_ > best_cv_score:
                            best_cv_score = grid.best_score_
                    ensemble_model = VotingRegressor(estimators=tuned_estimators)
                    best_model = ensemble_model
                    best_model_name = "VotingRegressor (Ensemble)"
                    st.session_state.best_model_name = best_model_name
            else:
                if model_choice == "Linear Regression":
                    best_model = LinearRegression()
                elif model_choice == "Logistic Regression":
                    best_model = LogisticRegression()
                elif model_choice == "Random Forest":
                    best_model = RandomForestClassifier() if is_classification else RandomForestRegressor()
                elif model_choice == "Gradient Boosting":
                    best_model = GradientBoostingClassifier() if is_classification else GradientBoostingRegressor()
                elif model_choice == "SVM":
                    best_model = SVC(probability=True)
                elif model_choice == "KNN":
                    best_model = KNeighborsClassifier()
                else:
                    st.error("Unknown model choice.")
                    st.stop()
                best_model_name = model_choice
                st.session_state.best_model_name = best_model_name

            best_model.fit(X_train, y_train)
            y_pred = best_model.predict(X_test)
            if not is_classification:
                test_score = abs(r2_score(y_test, y_pred)) * 100
                st.write(f"### {st.session_state.best_model_name} R² Score: {test_score:.2f}%")
            else:
                test_score = accuracy_score(y_test, y_pred) * 100
                st.write(f"### {st.session_state.best_model_name} Accuracy: {test_score:.2f}%")
            st.session_state.test_score = test_score
            st.session_state.best_model = best_model

        if st.session_state.best_model is not None:
            st.write(f"**Selected Model:** {st.session_state.best_model_name}")
            if st.session_state.is_classification and st.session_state.best_model_name != "Linear Regression":
                st.write(f"**Test Accuracy:** {st.session_state.test_score:.2f}%")
            else:
                st.write(f"**Test R² Score (as Accuracy %):** {st.session_state.test_score:.2f}%")
    if st.session_state.best_model_name and st.session_state.test_score:
        st.write(f"### Model in use: {st.session_state.best_model_name} ( Accuracy: {st.session_state.test_score:0.2f}%)")

    st.write("### Make a Prediction")
    user_input = {}
    if st.session_state.dataset_type == "Textual":
        combined_text = " ".join([st.text_input(f"Enter text for {feature}", value=str(df[feature].iloc[0]), key=f"{feature}_input") for feature in st.session_state.chosen_features])
        user_input["combined_text"] = combined_text.strip()
    else:
        for feature in st.session_state.chosen_features:
            if pd.api.types.is_numeric_dtype(df[feature]):
                default_val = float(df[feature].median())
                val = st.text_input(f"Enter {feature}", value=str(default_val), key=f"{feature}_input")
                try:
                    user_input[feature] = float(val)
                except ValueError:
                    user_input[feature] = default_val
            else:
                unique_vals = df[feature].unique().tolist()
                selected_val = st.selectbox(f"Select {feature}", options=unique_vals, key=f"{feature}_input")
                le = LabelEncoder()
                le.fit(df[feature])
                user_input[feature] = le.transform([selected_val])[0]

    if st.button("Predict"):
        if st.session_state.best_model is None or st.session_state.best_model == "dummy_model":
            st.error("Please train the model first!")
        else:
            if st.session_state.dataset_type == "Textual":
                from sklearn.feature_extraction.text import TfidfVectorizer
                tfidf = TfidfVectorizer()
                input_features = tfidf.fit_transform([user_input["combined_text"]])
            else:
                input_df = pd.DataFrame([user_input])
                if st.session_state.poly is not None:
                    input_poly = st.session_state.poly.transform(input_df)
                    input_features = st.session_state.scaler.transform(input_poly)
                else:
                    input_features = st.session_state.scaler.transform(input_df)
            prediction = st.session_state.best_model.predict(input_features)
            st.session_state.prediction_result = prediction[0]
            
if st.session_state.prediction_result is not None:
    st.write("### Prediction:", st.session_state.prediction_result)