# Importing necessary packages
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import streamlit as st

# SKLearn initialiation
from sklearn.model_selection import train_test_split

# Models
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import r2_score, accuracy_score

uploaded_file = st.file_uploader("Upload a CSV File", type=[ "csv", "tsv", "xlsx", "json", "parquet", "pkl", "h5", "xml" ])
if uploaded_file is not None:
    # Load the uploaded file
    df = pd.read_csv(uploaded_file)
    st.write("### Data Preview")
    st.write(df.head())

    num_features = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_features = df.select_dtypes(exclude=[np.number]).columns.tolist()

    