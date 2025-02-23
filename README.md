# Ez-Viz: ML Visualization & Prediction Tool
This project is a web-based tool built with Streamlit that allows users to visualize and predict outcomes from their datasets. It supports loading data from a CSV file or a URL, cleans the data, provides a range of visualization options, and trains an optimized machine learning model (either classification or regression). The tool includes dynamic input for predictions and allows users to export the trained model in their preferred format.

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
- Download the trained model in either **pickle** or **joblib** format.

# Prediction Output
For classification problems, the model outputs the predicted class along with a confidence score. Numeric predictions are mapped back to the original string labels.
For regression, the numeric prediction (with R² score) is displayed.

# Installation
### Prerequisites
Python 3.7 or higher

### Required Libraries
Install the required Python libraries using pip:
`pip install -r requirements.txt`

### Running the Application
To run the application on local machine, navigate to the project directory in your terminal and run:
`streamlit run Ez-Viz.py`

# Site hosted on: https://ez-viz.streamlit.app

# Prediction Model

Select (or override) the target variable from your dataset.
The model is trained (or retrained) automatically based on the chosen target.
Use the dynamic input section to modify predictors and make predictions.
For classification, the model outputs the predicted class with a confidence score. For regression, the predicted numeric value is displayed.

# Export Trained Model
In the sidebar, select your preferred export format (pickle or joblib) and download the trained model.

# Troubleshooting
### Long Training Time:
If the model takes too long to train, consider reducing the hyperparameter grid or using a smaller dataset.

### Negative R² Scores:
A negative R² indicates that the model performs worse than a baseline prediction. Try adjusting feature selection or tuning model parameters.

### Data URL Issues:
If there are errors loading data from a URL, ensure the URL points to a valid CSV file. The app uses the Python engine with on_bad_lines='skip' to mitigate tokenizing errors.

# Contributing
Contributions are welcome! Feel free to fork this repository and submit pull requests. For major changes, please open an issue first to discuss what you would like to change.

# License
This project is licensed under the MIT License.

~ made by Team: Schrödinger's Cat
