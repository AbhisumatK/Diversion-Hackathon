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
from sklearn.metrics import accuracy_score, r2_score

st.title("Machine Learning Model Trainer")

# File Upload
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### Data Preview")
    st.write(df.head())
    
    # Identifying numerical and categorical features
    num_features = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_features = df.select_dtypes(exclude=[np.number]).columns.tolist()
    
    st.write("### Column Analysis")
    st.write("**Numerical Features:**", num_features)
    st.write("**Categorical Features:**", cat_features)
    
    # Correlation matrix
    st.write("### Correlation Matrix")
    corr_matrix = df[num_features].corr()
    fig, ax = plt.subplots()
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)
    
    # Selecting target variable
    target_column = st.selectbox("Select Target Column", df.columns)
    
    # Dropping columns with near-zero correlation
    if target_column in num_features:
        target_corr = corr_matrix[target_column].abs()
        selected_features = target_corr[target_corr > 0.1].index.tolist()
    else:
        selected_features = num_features  # Use all numerical features for classification
    
    if target_column in selected_features:
        selected_features.remove(target_column)
    
    st.write("### Selected Features", selected_features)
    
    # Feature Selection
    chosen_features = st.multiselect("Choose Features for Training", selected_features, default=selected_features)
    
    # Data Preprocessing
    X = df[chosen_features]
    y = df[target_column]
    
    # Encoding categorical target if necessary
    if y.dtype == 'O':
        le = LabelEncoder()
        y = le.fit_transform(y)
    
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Model Selection
    model_choice = st.selectbox("Choose Model", ["Logistic Regression", "Random Forest", "Gradient Boosting", "SVM", "KNN", "Linear Regression"])
    
    # Suggest Best Models based on Feature Criteria
    best_models = []
    if len(selected_features) < 5:
        best_models.append("KNN")
    if len(selected_features) > 5:
        best_models.append("Random Forest")
    if target_column in cat_features or y.nunique() < 10:
        best_models.extend(["Logistic Regression", "Gradient Boosting", "SVM"])
    else:
        best_models.append("Linear Regression")
    
    best_models = list(set(best_models))
    st.write("### Recommended Models", best_models)
    
    if st.button("Train Model"):
        if model_choice == "Logistic Regression":
            model = LogisticRegression()
        elif model_choice == "Random Forest":
            model = RandomForestClassifier()
        elif model_choice == "Gradient Boosting":
            model = GradientBoostingClassifier()
        elif model_choice == "SVM":
            model = SVC()
        elif model_choice == "KNN":
            model = KNeighborsClassifier()
        elif model_choice == "Linear Regression":
            model = LinearRegression()
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        if model_choice == "Linear Regression":
            score = r2_score(y_test, y_pred) * 100  # Convert R^2 score to percentage
            st.write(f"### Model R² Score: {score:.2f}%")
        else:
            score = accuracy_score(y_test, y_pred) * 100
            st.write(f"### Model Accuracy: {score:.2f}%")
    
    # Visualization Options
    st.write("### Data Visualizations")
    viz_option = st.selectbox("Choose a Visualization", ["Pie Chart", "Box Plot", "Error Bar", "Scatter Plot", "Area Under Curve", "Histogram", "Violin Plot"])
    
    if viz_option == "Pie Chart" and target_column in cat_features:
        fig, ax = plt.subplots()
        df[target_column].value_counts().plot.pie(autopct='%1.1f%%', ax=ax)
        st.pyplot(fig)
    elif viz_option == "Box Plot":
        fig, ax = plt.subplots()
        sns.boxplot(data=df[chosen_features], ax=ax)
        st.pyplot(fig)
    elif viz_option == "Error Bar":
        fig, ax = plt.subplots()
        means = df[chosen_features].mean()
        stds = df[chosen_features].std()
        ax.errorbar(means.index, means, yerr=stds, fmt='o', capsize=5)
        st.pyplot(fig)
    elif viz_option == "Scatter Plot":
        feature_x = st.selectbox("Select X-axis feature", chosen_features)
        feature_y = st.selectbox("Select Y-axis feature", chosen_features)
        fig, ax = plt.subplots()
        ax.scatter(df[feature_x], df[feature_y])
        ax.set_xlabel(feature_x)
        ax.set_ylabel(feature_y)
        st.pyplot(fig)
    elif viz_option == "Area Under Curve" and model_choice != "Linear Regression":
        from sklearn.metrics import roc_curve, auc
        if len(set(y)) == 2:  # Binary classification only
            y_proba = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            auc_score = auc(fpr, tpr)
            fig, ax = plt.subplots()
            ax.plot(fpr, tpr, label=f'AUC = {auc_score:.2f}')
            ax.plot([0, 1], [0, 1], 'k--')
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.legend()
            st.pyplot(fig)
        else:
            st.write("AUC visualization is only available for binary classification.")