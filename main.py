# Importing necessary packages
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import streamlit as st

# SKLearn initialiation
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
import models

uploaded_file = st.file_uploader("Upload a CSV File", type=[ "csv", "tsv", "xlsx", "json", "parquet", "pkl", "h5", "xml" ])
if uploaded_file is not None:
    # Load the uploaded file
    df = pd.read_csv(uploaded_file)
    st.write("### Data Preview")
    st.write(df.head())

    num_features = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_features = df.select_dtypes(exclude=[np.number]).columns.tolist()

<<<<<<< Updated upstream
<<<<<<< Updated upstream
<<<<<<< Updated upstream
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

    
=======
>>>>>>> Stashed changes
=======
>>>>>>> Stashed changes
=======
>>>>>>> Stashed changes
