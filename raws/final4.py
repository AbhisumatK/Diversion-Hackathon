import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import StackingRegressor, RandomForestRegressor, GradientBoostingRegressor
from sklearn.ensemble import StackingClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report

st.title("Machine Learning Visualization & Prediction Tool")

# -------------------------------
# Step 1: Dataset Upload and Preview
# -------------------------------
st.sidebar.header("Upload Your CSV File")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### Dataset Preview")
    st.write(df.head())

    # -------------------------------
    # Step 2: Data Cleaning
    # -------------------------------
    df.drop_duplicates(inplace=True)
    df.fillna(df.median(numeric_only=True), inplace=True)
    for col in df.select_dtypes(include=['object']).columns:
        df[col].fillna(df[col].mode()[0], inplace=True)

    # -------------------------------
    # Step 3: Column Analysis
    # -------------------------------
    st.write("### Column Analysis")
    numerical_cols = df.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=['number']).columns.tolist()
    st.write("Numerical Columns:", numerical_cols)
    st.write("Categorical Columns:", categorical_cols)

    # -------------------------------
    # Step 4: Description of Data
    # -------------------------------
    st.write("### Summary Statistics")
    st.write(df[numerical_cols].describe())

    # Encode categorical variables for visualization.
    # Save the encoders to later map predictions back to original strings.
    label_encoders = {}
    for col in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    # -------------------------------
    # Data Filtering
    # -------------------------------
    if len(numerical_cols) > 2:
        a = numerical_cols[1]
        b = numerical_cols[2]
        st.write("### Data Filtering")
        filter_conditions = st.text_area(
            f"Enter filter conditions (e.g., {a} > {df[a].values[1]} & {b} < {df[b].values[3]})"
        )
        if filter_conditions:
            try:
                df = df.query(filter_conditions)
                st.write("Filtered Dataset Preview")
                st.write(df.head())
            except Exception as e:
                st.error(f"Invalid filter condition: {e}")

    # -------------------------------
    # Normalization / Scaling for Visualization
    # -------------------------------
    st.write("### Normalize/Scale Numerical Values")
    if st.checkbox("Normalize/Scale Numerical Columns"):
        scaler = StandardScaler()
        df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
        st.write("Normalized Dataset Preview")
        st.write(df.head())

    # -------------------------------
    # Step 5: Data Visualizations
    # -------------------------------
    st.write("### Data Visualizations")
    suggested_charts = []
    if len(numerical_cols) >= 2:
        suggested_charts.extend(["Line", "Scatter", "Bar", "Box", "Violin"])
    if len(categorical_cols) > 0:
        suggested_charts.extend(["Pie", "Bar"])
    if len(numerical_cols) == 1:
        suggested_charts.append("Histogram")

    st.write("### Suggested Charts:", ', '.join(suggested_charts))
    chart_type = st.selectbox("Select Chart Type", ["Bar", "Line", "Scatter", "Histogram", "Box", "Violin", "Pie"])

    if chart_type not in suggested_charts:
        st.warning("If you choose graphs outside the preference, the depiction may result in ambiguity.")

    x_axis = st.selectbox("Select X-axis", df.columns)

    if chart_type == "Line":
        y_axes = st.multiselect("Select Y-axis (Multiple for Line Chart)", df.columns)
        legend_position = st.selectbox("Select Legend Position", ["Top Right", "Top Left", "Bottom Right", "Bottom Left"])
        legend_positions = {
            "Top Right": dict(x=1, y=1),
            "Top Left": dict(x=0, y=1),
            "Bottom Right": dict(x=1, y=0),
            "Bottom Left": dict(x=0, y=0)
        }
    else:
        y_axis = st.selectbox("Select Y-axis", df.columns if chart_type not in ["Pie"] else [None])

    if chart_type == "Bar":
        fig = px.bar(df, x=x_axis, y=y_axis, title=f"Bar Chart of {x_axis} vs {y_axis}", barmode='group', hover_data=df.columns)
    elif chart_type == "Line":
        if len(y_axes) > 0:
            fig = go.Figure()
            for y in y_axes:
                fig.add_trace(go.Scatter(x=df[x_axis], y=df[y], mode='lines', name=y))
            fig.update_layout(title=f"Line Chart: {x_axis} vs Multiple Y-axes", xaxis_title=x_axis, yaxis_title="Values", legend=legend_positions[legend_position])
        else:
            st.error("Please select at least one Y-axis for the Line chart.")
    elif chart_type == "Scatter":
        fig = px.scatter(df, x=x_axis, y=y_axis, title=f"Scatter Plot of {x_axis} vs {y_axis}", color=df[x_axis], size_max=10, hover_data=df.columns)
    elif chart_type == "Histogram":
        fig = px.histogram(df, x=x_axis, title=f"Histogram of {x_axis}", marginal="box", hover_data=df.columns)
    elif chart_type == "Box":
        fig = px.box(df, x=x_axis, y=y_axis, title=f"Box Plot of {x_axis} vs {y_axis}", points="all", hover_data=df.columns)
    elif chart_type == "Violin":
        fig = px.violin(df, x=x_axis, y=y_axis, title=f"Violin Plot of {x_axis} vs {y_axis}", box=True, points="all", hover_data=df.columns)
    elif chart_type == "Pie":
        pie_values = df[x_axis].value_counts().reset_index()
        pie_values.columns = [x_axis, 'Count']
        fig = px.pie(pie_values, names=x_axis, values='Count', title=f"Pie Chart of {x_axis}", hover_data=['Count'])

    fig.update_layout(hovermode='x unified')
    st.plotly_chart(fig)

    # ---------------------------------------
    # Prediction Model Section - Enhanced with Optimized Pipelines & Dynamic Input
    # ---------------------------------------
    st.write("## Prediction Model")
    st.info("This advanced prediction model uses a stacking ensemble with optimized preprocessing. "
            "A ColumnTransformer first applies scaling (StandardScaler) on numeric features and mapping (OneHotEncoder) on categorical features before fitting the model. "
            "The target variable is auto-selected, and the appropriate model (classification or regression) is used. "
            "For dynamic input: if all predictors are categorical, you'll provide input for every feature; "
            "otherwise, you'll modify one feature at a time.")

    # AUTOMATED TARGET SELECTION:
    if "target" in df.columns:
        target_col = "target"
    else:
        target_col = df.columns[-1]
    st.write("Auto-selected target variable for prediction:", target_col)

    # Prepare predictors (X) and target (y)
    X = df.drop(target_col, axis=1)
    y = df[target_col]

    # Save training columns for consistency.
    if "training_columns" not in st.session_state:
        st.session_state["training_columns"] = list(X.columns)

    # Identify numeric and categorical predictors
    num_cols = X.select_dtypes(include=np.number).columns.tolist()
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()

    # Build a ColumnTransformer to preprocess features (scaling + mapping)
    transformers = []
    if num_cols:
        transformers.append(("num", StandardScaler(), num_cols))
    if cat_cols:
        transformers.append(("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols))
    preprocessor = ColumnTransformer(transformers=transformers)

    # AUTOMATED MODEL SELECTION: classification if target is object or few unique values; else regression.
    is_classification = (y.dtype == "object") or (y.nunique() < 10)

    # Train the model if not already stored or if target has changed.
    if ("trained_model" not in st.session_state) or (st.session_state.get("target_col") != target_col):
        st.write("Training optimized model...")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        if is_classification:
            st.write("Detected Classification Problem")
            estimators = [
                ('rf', RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)),
                ('gbc', GradientBoostingClassifier(n_estimators=100, random_state=42))
            ]
            model = StackingClassifier(
                estimators=estimators,
                final_estimator=LogisticRegression(),
                n_jobs=-1
            )
            pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])
            param_grid = {
                "model__rf__n_estimators": [100, 200],
                "model__gbc__n_estimators": [100, 200]
            }
            grid = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1)
            grid.fit(X_train, y_train)
            best_model = grid.best_estimator_
            y_pred = best_model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            st.write("Best Hyperparameters:", grid.best_params_)
            st.write("Accuracy:", acc)
            st.session_state["model_metrics"] = {"Accuracy": acc}
        else:
            st.write("Detected Regression Problem")
            estimators = [
                ('rf', RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)),
                ('gbr', GradientBoostingRegressor(n_estimators=100, random_state=42))
            ]
            model = StackingRegressor(
                estimators=estimators,
                final_estimator=Ridge(),
                n_jobs=-1
            )
            pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])
            param_grid = {
                "model__rf__n_estimators": [100, 200],
                "model__gbr__n_estimators": [100, 200]
            }
            grid = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1)
            grid.fit(X_train, y_train)
            best_model = grid.best_estimator_
            y_pred = best_model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            st.write("Best Hyperparameters:", grid.best_params_)
            st.write("Mean Squared Error:", mse)
            st.write("R² Score:", r2)
            st.session_state["model_metrics"] = {"MSE": mse, "R2": r2}
        st.session_state["trained_model"] = best_model
        st.session_state["target_col"] = target_col
    else:
        best_model = st.session_state["trained_model"]
        st.write("Loaded trained model from session state.")
        st.write("Model Metrics:", st.session_state["model_metrics"])

    # -------------------------------
    # Dynamic Input Section for Prediction
    # -------------------------------
    st.write("### Dynamic Prediction Input")
    st.info("If all predictors are categorical, you'll provide input for every feature. Otherwise, select one feature to modify (others use default values: median for numeric, mode for categorical).")

    # Initialize default input values in session state using stored training columns.
    if "input_values" not in st.session_state:
        st.session_state["input_values"] = {}
        for col in st.session_state["training_columns"]:
            if col in num_cols:
                st.session_state["input_values"][col] = X[col].median()
            else:
                st.session_state["input_values"][col] = X[col].mode()[0]

    # If all predictors are categorical, prompt input for each feature.
    if len(num_cols) == 0:
        st.write("**All predictors are categorical. Please select a value for each feature:**")
        input_vector = {}
        for col in st.session_state["training_columns"]:
            unique_vals = sorted(X[col].unique())
            input_vector[col] = st.selectbox(f"Select value for '{col}'", options=unique_vals, key=f"input_{col}")
    else:
        # Otherwise, allow selection of one feature to modify.
        selected_feature = st.selectbox("Select Feature to Modify", options=st.session_state["training_columns"], key="custom_feature_select")
        if selected_feature:
            if selected_feature in num_cols:
                new_val = st.number_input(
                    f"Enter custom value for '{selected_feature}'",
                    value=float(st.session_state["input_values"][selected_feature]),
                    key=f"input_{selected_feature}"
                )
            else:
                unique_vals = sorted(X[selected_feature].unique())
                default_index = unique_vals.index(st.session_state["input_values"][selected_feature]) if st.session_state["input_values"][selected_feature] in unique_vals else 0
                new_val = st.selectbox(
                    f"Select custom value for '{selected_feature}'",
                    options=unique_vals,
                    index=default_index,
                    key=f"input_{selected_feature}"
                )
            st.session_state["input_values"][selected_feature] = new_val
            input_vector = {col: st.session_state["input_values"].get(col) for col in st.session_state["training_columns"]}

    st.write("**Current Input Vector for Prediction:**", input_vector)

    # Predict using the trained model pipeline.
    input_df = pd.DataFrame([input_vector])
    prediction = best_model.predict(input_df)
    
    # If classification, map numeric prediction back to original string using the saved LabelEncoder.
    if is_classification and target_col in label_encoders:
        prediction_mapped = label_encoders[target_col].inverse_transform([prediction[0]])[0]
        st.write("### Prediction Result:", prediction_mapped)
    else:
        st.write("### Prediction Result:", prediction[0])
