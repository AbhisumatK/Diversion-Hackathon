# Machine Learning Visualization & Prediction Tool
This project is a web-based tool built with Streamlit that allows users to visualize and predict outcomes from their datasets. It supports loading data from a CSV file or a URL, cleans the data, provides a range of visualization options, and trains an optimized machine learning model (either classification or regression). The tool includes dynamic input for predictions and allows users to export the trained model in their preferred format.

# Features
Dataset Loading

Load datasets by uploading a CSV file.
Alternatively, extract the dataset from a URL.
Data Cleaning & Analysis

Automatic duplicate removal and missing value imputation (median for numeric, mode for categorical).
Provides column analysis and summary statistics.
Data Visualization

Multiple chart types including Bar, Line, Scatter, Histogram, Box, Violin, and Pie charts.
Customizable visualization parameters (e.g., selecting axes, legend positions).
Dynamic Prediction Model

Automatically selects a classification or regression model based on the target variable.
Uses a stacking ensemble with optimized preprocessing (scaling for numeric and one-hot encoding for categorical features).
Hyperparameter tuning via GridSearchCV.
Retrains the model if the dataset or target variable changes.
Provides dynamic input where:
For all-categorical predictors: input for every feature.
Otherwise: modification of one feature at a time (with immediate state update via callbacks).
Prediction Output

For classification problems, the model outputs the predicted class along with a confidence score. Numeric predictions are mapped back to the original string labels.
For regression, the numeric prediction (with R² score) is displayed.
Model Export

Export the trained model from the sidebar in either pickle or joblib format.
Installation
Prerequisites
Python 3.7 or higher

Required Libraries
Install the required Python libraries using pip:
`pip install streamlit pandas numpy plotly scikit-learn joblib`

Running the Application
To run the application, navigate to the project directory in your terminal and run:
`streamlit run ez-viz.py`

# Usage
Load Your Dataset
Use the sidebar to either upload a CSV file or enter a URL to load your dataset.

Data Cleaning & Visualization
The app automatically cleans the data and displays summary statistics. You can also choose various visualization options.

Prediction Model

Select (or override) the target variable from your dataset.
The model is trained (or retrained) automatically based on the chosen target.
Use the dynamic input section to modify predictors and make predictions.
For classification, the model outputs the predicted class with a confidence score. For regression, the predicted numeric value is displayed.
Export Trained Model
In the sidebar, select your preferred export format (pickle or joblib) and download the trained model.

# Troubleshooting
Long Training Time:
If the model takes too long to train, consider reducing the hyperparameter grid or using a smaller dataset.

Negative R² Scores:
A negative R² indicates that the model performs worse than a baseline prediction. Try adjusting feature selection or tuning model parameters.

Data URL Issues:
If there are errors loading data from a URL, ensure the URL points to a valid CSV file. The app uses the Python engine with on_bad_lines='skip' to mitigate tokenizing errors.

# Contributing
Contributions are welcome! Feel free to fork this repository and submit pull requests. For major changes, please open an issue first to discuss what you would like to change.

# License
This project is licensed under the MIT License.

~ made by Team: Schrödinger's Cat
