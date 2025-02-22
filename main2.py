import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, r2_score, roc_curve, auc

st.set_page_config(layout="wide")
st.title("Machine Learning Model Trainer")

# Initialize session state variables if not already present
if "df" not in st.session_state:
    st.session_state.df = None
if "best_model" not in st.session_state:
    st.session_state.best_model = None
if "scaler" not in st.session_state:
    st.session_state.scaler = None
if "chosen_features" not in st.session_state:
    st.session_state.chosen_features = None
if "is_classification" not in st.session_state:
    st.session_state.is_classification = None
if "best_model_name" not in st.session_state:
    st.session_state.best_model_name = None
if "test_score" not in st.session_state:
    st.session_state.test_score = None

# Sidebar: File upload and options
with st.sidebar:
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    prediction_mode = st.radio("Prediction Mode", ["Manual Input", "Upload CSV"])
    if uploaded_file:
        if st.session_state.df is None:
            df = pd.read_csv(uploaded_file)
            df.dropna(inplace=True)
            st.session_state.df = df
        else:
            df = st.session_state.df

        st.write("### Column Analysis")
        num_features = df.select_dtypes(include=[np.number]).columns.tolist()
        cat_features = df.select_dtypes(exclude=[np.number]).columns.tolist()
        st.write("**Numerical Features:**", num_features)
        st.write("**Categorical Features:**", cat_features)

        target_column = st.selectbox("Select Target Column", df.columns)
        chosen_features = st.multiselect("Choose Features for Training", num_features, default=num_features)
        st.session_state.chosen_features = chosen_features

        viz_option = st.selectbox("Choose a Visualization", 
                                  ["Pie Chart", "Box Plot", "Error Bar", "Scatter Plot", 
                                   "Area Under Curve", "Histogram", "Violin Plot"])
        
        # If prediction mode is Upload CSV, allow a file upload for new prediction data.
        if prediction_mode == "Upload CSV":
            pred_file = st.file_uploader("Upload CSV for Prediction", type=["csv"])
            st.session_state.pred_file = pred_file
        else:
            st.session_state.pred_file = None

if st.session_state.df is not None:
    df = st.session_state.df
    st.write("### Data Preview")
    st.write(df.head())

    # Correlation Matrix
    st.write("### Correlation Matrix")
    num_features = df.select_dtypes(include=[np.number]).columns.tolist()
    corr_matrix = df[num_features].corr()
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    # Data Preprocessing
    X = df[st.session_state.chosen_features]
    y = df[target_column]

    # Determine problem type
    if (df[target_column].dtype in [np.int64, np.float64]) and (df[target_column].nunique() > 10):
        is_classification = False
    else:
        is_classification = True
    st.session_state.is_classification = is_classification

    # Encode target if needed
    if is_classification and y.dtype == 'O':
        le = LabelEncoder()
        y = le.fit_transform(y)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    st.session_state.scaler = scaler

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    st.write("### Auto Train & Predict")
    if st.button("Auto Train & Predict"):
        # Define candidate models based on problem type
        if is_classification:
            candidate_models = {
                "Logistic Regression": LogisticRegression(),
                "Random Forest": RandomForestClassifier(),
                "Gradient Boosting": GradientBoostingClassifier(),
                "SVM": SVC(probability=True),
                "KNN": KNeighborsClassifier()
            }
            scoring = 'accuracy'
        else:
            candidate_models = {"Linear Regression": LinearRegression()}
            scoring = 'r2'

        # Evaluate candidates using cross-validation
        scores = {}
        for name, model in candidate_models.items():
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring=scoring)
            scores[name] = np.mean(cv_scores)
            st.write(f"**{name} CV Score:** {scores[name]*100:.2f}%")

        # Automatically select the best model
        best_model_name = max(scores, key=scores.get)
        best_model = candidate_models[best_model_name]
        st.write(f"### Best Model Selected: {best_model_name}")
        st.session_state.best_model_name = best_model_name

        # Train best model on the full training set
        best_model.fit(X_train, y_train)
        y_pred = best_model.predict(X_test)
        if is_classification:
            test_score = accuracy_score(y_test, y_pred) * 100
            st.write(f"### Test Accuracy: {test_score:.2f}%")
        else:
            test_score = r2_score(y_test, y_pred) * 100
            st.write(f"### Test R² Score (as Accuracy %): {test_score:.2f}%")
        st.session_state.test_score = test_score
        st.session_state.best_model = best_model

    # Display best model details if already trained
    if st.session_state.best_model is not None:
        st.write(f"**Best Model:** {st.session_state.best_model_name}")
        if st.session_state.is_classification:
            st.write(f"**Test Accuracy:** {st.session_state.test_score:.2f}%")
        else:
            st.write(f"**Test R² Score (as Accuracy %):** {st.session_state.test_score:.2f}%")

    st.write("### Make a Prediction")
    if prediction_mode == "Manual Input":
        # Manual input mode: create one input per feature
        user_input = {}
        for feature in st.session_state.chosen_features:
            default_val = float(df[feature].median())
            val = st.text_input(f"Enter {feature}", value=str(default_val), key=f"{feature}_input")
            try:
                user_input[feature] = float(val)
            except ValueError:
                user_input[feature] = default_val

        if st.button("Predict"):
            if st.session_state.best_model is None:
                st.error("Please train the model first!")
            else:
                input_df = pd.DataFrame([user_input])
                input_df = pd.DataFrame(st.session_state.scaler.transform(input_df), columns=st.session_state.chosen_features)
                prediction = st.session_state.best_model.predict(input_df)
                st.write(f"### Prediction: {prediction[0]}")
    else:
        # CSV upload mode for prediction: if a prediction CSV was uploaded
        if st.session_state.pred_file is not None:
            pred_df = pd.read_csv(st.session_state.pred_file)
            st.write("### Prediction Input Data")
            st.write(pred_df.head())
            # Use only the chosen features (if they exist in pred_df)
            missing_features = set(st.session_state.chosen_features) - set(pred_df.columns)
            if missing_features:
                st.error(f"The following features are missing in the prediction file: {missing_features}")
            else:
                pred_df_scaled = pd.DataFrame(st.session_state.scaler.transform(pred_df[st.session_state.chosen_features]),
                                              columns=st.session_state.chosen_features)
                prediction = st.session_state.best_model.predict(pred_df_scaled)
                pred_df['Predicted_Target'] = prediction
                st.write("### Predictions:")
                st.write(pred_df)
        else:
            st.info("Please upload a CSV file for prediction in the sidebar.")

    # Data Visualizations
    st.write("### Data Visualizations")
    if viz_option == "Pie Chart" and target_column in cat_features:
        fig, ax = plt.subplots()
        df[target_column].value_counts().plot.pie(autopct='%1.1f%%', ax=ax)
        st.pyplot(fig)
    elif viz_option == "Box Plot":
        fig, ax = plt.subplots()
        sns.boxplot(data=df[st.session_state.chosen_features], ax=ax)
        st.pyplot(fig)
    elif viz_option == "Error Bar":
        fig, ax = plt.subplots()
        means = df[st.session_state.chosen_features].mean()
        stds = df[st.session_state.chosen_features].std()
        ax.errorbar(means.index, means, yerr=stds, fmt='o', capsize=5)
        st.pyplot(fig)
    elif viz_option == "Scatter Plot":
        with st.sidebar:
            feature_x = st.selectbox("Select X-axis feature", st.session_state.chosen_features)
            feature_y = st.selectbox("Select Y-axis feature", st.session_state.chosen_features)
        fig, ax = plt.subplots()
        ax.scatter(df[feature_x], df[feature_y])
        ax.set_xlabel(feature_x)
        ax.set_ylabel(feature_y)
        st.pyplot(fig)
    elif viz_option == "Area Under Curve" and st.session_state.is_classification and len(set(y)) == 2 and st.session_state.best_model:
        y_proba = st.session_state.best_model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc_score = auc(fpr, tpr)
        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, label=f'AUC = {auc_score:.2f}')
        ax.plot([0, 1], [0, 1], 'k--')
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.legend()
        st.pyplot(fig)
    elif viz_option == "Histogram":
        fig, ax = plt.subplots()
        df[st.session_state.chosen_features].hist(ax=ax)
        st.pyplot(fig)
    elif viz_option == "Violin Plot":
        fig, ax = plt.subplots()
        sns.violinplot(data=df[st.session_state.chosen_features], ax=ax)
        st.pyplot(fig)