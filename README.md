# Ez-Viz: ML Visualization & Prediction Tool
This project is a web-based tool built with Streamlit that allows users to visualize and predict outcomes from their datasets. It supports loading data from a CSV file or a URL, cleans the data, provides a range of visualization options, and trains an optimized machine learning model (either classification or regression). The tool includes dynamic input for predictions and allows users to export the trained model in their preferred format. 
# Demo video: https://youtu.be/ZZ8xRK2R578
# Features
✅ **Load Datasets**
- Upload a CSV file or extract data from a URL.
✅ **Data Cleaning & Analysis**
- Automatically removes duplicates.
- Fills missing values (median for numbers, mode for categories).
- Provides column analysis and summary statistics.
✅ **Data Visualization**
- Choose from multiple chart types: Bar, Line, Scatter, Histogram, Box, Violin, and Pie.
- Customize visualization parameters (e.g., axis selection, legend positioning).
✅ **Dynamic Machine Learning Model**
- Automatically detects and applies classification or regression.
- Uses advanced techniques like stacking ensemble and GridSearchCV for optimization.
- Retrains the model when dataset or target variable changes.
✅ **Interactive Prediction**
- Provides dynamic input forms:
  - For categorical features: input for each feature.
  - For mixed data: modify one feature at a time.
✅ **Export Trained Model**
- Download the trained model in either pickle or joblib format.

# Prediction Output
For classification problems, the model outputs the predicted class along with a confidence score. Numeric predictions are mapped back to the original string labels. For regression, the numeric prediction (with R² score) is displayed.

# Installation

### Prerequisites
Python 3.7 or higher

### Required Libraries
Install the required Python libraries using pip:
```
pip install -r requirements.txt
```

### Running the Application
To run the application on local machine, navigate to the project directory in your terminal and run:
```
streamlit run Ez-Viz.py
```

# Site hosted on: https://ez-viz.streamlit.app 
