import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, r2_score
import numpy as np

st.title("Machine Learning Model Trainer")

# Step 1: Dataset Upload
st.sidebar.header("Upload Your CSV File")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### Dataset Preview")
    st.write(df.head())
    
    # Step 2: Column Analysis
    st.write("### Column Analysis")
    numerical_cols = df.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=['number']).columns.tolist()
    st.write("Numerical Columns:", numerical_cols)
    st.write("Categorical Columns:", categorical_cols)
    
    # Encode categorical variables
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
    
    # Step 3: Correlation Matrix
    if st.checkbox("Show Correlation Matrix"):
        st.write("#### Correlation Matrix")
        corr_matrix = df.corr()
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
        st.pyplot(fig)
        
        # Automatically select features with high correlation to the target
        target_corr = corr_matrix.abs().iloc[:, -1].sort_values(ascending=False)
        recommended_features = target_corr[target_corr > 0.1].index.tolist()
        if len(recommended_features) > 1 and target_corr.index[-1] in recommended_features:
            recommended_features.remove(target_corr.index[-1])  # Remove the target itself if present
        else:
            recommended_features = []  # Drop features with near-zero correlation
    else:
        recommended_features = numerical_cols
    
    # Step 4: Feature Selection
    st.sidebar.header("Feature Selection")
    selected_features = st.sidebar.multiselect("Select Features", recommended_features)
    target = st.sidebar.selectbox("Select Target Variable", df.columns)
    
    if selected_features and target:
        X = df[selected_features]
        y = df[target]
        
        if y.nunique() <= 1:
            st.error("Target variable must have more than one unique value.")
        else:
            # Step 5: Scaling
            scaler = StandardScaler()
            X = pd.DataFrame(scaler.fit_transform(X), columns=selected_features)
            
            # Step 6: Model Training
            st.sidebar.header("Model Selection")
            model_choice = st.sidebar.selectbox("Choose Model", [
                "Linear Regression", "Logistic Regression", "Decision Tree", "Random Forest", "Gradient Boosting", "AdaBoost", "SVM", "K-Nearest Neighbors"
            ])
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            if model_choice == "Linear Regression":
                model = LinearRegression()
                is_classification = False
            elif model_choice == "Logistic Regression":
                model = LogisticRegression(max_iter=500)
                is_classification = True
            elif model_choice == "Decision Tree":
                model = DecisionTreeClassifier()
                is_classification = True
            elif model_choice == "Random Forest":
                model = RandomForestClassifier()
                is_classification = True
            elif model_choice == "Gradient Boosting":
                model = GradientBoostingClassifier()
                is_classification = True
            elif model_choice == "AdaBoost":
                model = AdaBoostClassifier()
                is_classification = True
            elif model_choice == "SVM":
                model = SVC()
                is_classification = True
            else:
                model = KNeighborsClassifier()
                is_classification = True
            
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            if is_classification:
                metric = accuracy_score(y_test, y_pred)
                st.write(f"### Model Accuracy: {metric:.2f}")
            else:
                metric = r2_score(y_test, y_pred)
                st.write(f"### Model Accuracy: {metric * 100:.2f}%")
            
            # Step 7: Prediction Interface
            st.write("### Make a Prediction")
            user_input = {}
            for feature in selected_features:
                min_val = float(df[feature].min())
                max_val = float(df[feature].max())
                default_val = float(df[feature].median())
                user_input[feature] = st.number_input(f"Enter {feature}", min_value=min_val, max_value=max_val, value=default_val)
            
            if st.button("Predict"):
                input_df = pd.DataFrame([user_input])
                input_df = pd.DataFrame(scaler.transform(input_df), columns=selected_features)
                prediction = model.predict(input_df)
                st.write(f"### Prediction: {prediction[0]}")
            
            # Step 8: Visualizations
            st.write("### Data Visualizations")
            if st.checkbox("Show Feature Distributions"):
                st.write("#### Distribution of Features")
                fig, ax = plt.subplots(figsize=(10, 5))
                df[selected_features].hist(ax=ax)
                st.pyplot(fig)
            
            if hasattr(model, "feature_importances_"):
                st.write("#### Feature Importance")
                importance_df = pd.DataFrame({"Feature": selected_features, "Importance": model.feature_importances_})
                importance_df = importance_df.sort_values(by="Importance", ascending=False)
                fig, ax = plt.subplots()
                sns.barplot(x="Importance", y="Feature", data=importance_df, ax=ax)
                st.pyplot(fig)