import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import LabelEncoder, StandardScaler

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
    # Step 4: Data Description
    # -------------------------------
    st.write("### Summary Statistics")
    st.write(df[numerical_cols].describe())

    # Encode categorical variables (all become numeric)
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    # -------------------------------
    # Data Filtering
    # -------------------------------
    if len(numerical_cols) >= 3:
        a = numerical_cols[1]
        b = numerical_cols[2]
        st.write("### Data Filtering")
        filter_conditions = st.text_area(
            f"Enter filter conditions (`e.g., {a} > {df[a].values[1]} & {b} < {df[b].values[3]}`)"
        )
        if filter_conditions:
            try:
                df = df.query(filter_conditions)
                st.write("Filtered Dataset Preview")
                st.write(df.head())
            except Exception as e:
                st.error(f"Invalid filter condition: {e}")

    # -------------------------------
    # Normalization / Scaling
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
        # Legend Position for Line Chart
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
        fig = px.bar(df, x=x_axis, y=y_axis,
                     title=f"Bar Chart of {x_axis} vs {y_axis}",
                     barmode='group', hover_data=df.columns)
    elif chart_type == "Line":
        if len(y_axes) > 0:
            fig = go.Figure()
            for y in y_axes:
                fig.add_trace(go.Scatter(x=df[x_axis], y=df[y], mode='lines', name=y))
            fig.update_layout(title=f"Line Chart: {x_axis} vs Multiple Y-axes",
                              xaxis_title=x_axis, yaxis_title="Values",
                              legend=legend_positions[legend_position])
        else:
            st.error("Please select at least one Y-axis for the Line chart.")
    elif chart_type == "Scatter":
        fig = px.scatter(df, x=x_axis, y=y_axis,
                         title=f"Scatter Plot of {x_axis} vs {y_axis}",
                         color=df[x_axis], size_max=10, hover_data=df.columns)
    elif chart_type == "Histogram":
        fig = px.histogram(df, x=x_axis,
                           title=f"Histogram of {x_axis}",
                           marginal="box", hover_data=df.columns)
    elif chart_type == "Box":
        fig = px.box(df, x=x_axis, y=y_axis,
                     title=f"Box Plot of {x_axis} vs {y_axis}",
                     points="all", hover_data=df.columns)
    elif chart_type == "Violin":
        fig = px.violin(df, x=x_axis, y=y_axis,
                        title=f"Violin Plot of {x_axis} vs {y_axis}",
                        box=True, points="all", hover_data=df.columns)
    elif chart_type == "Pie":
        pie_values = df[x_axis].value_counts().reset_index()
        pie_values.columns = [x_axis, 'Count']
        fig = px.pie(pie_values, names=x_axis, values='Count',
                     title=f"Pie Chart of {x_axis}",
                     hover_data=['Count'])
    fig.update_layout(hovermode='x unified')
    st.plotly_chart(fig)

    # # -------------------------------
    # # Prediction Model Extension
    # # -------------------------------

    # -------------------------------
# Enhanced Prediction Model using Stacking Ensemble (Built-in AI)
# -------------------------------
    st.write("### Prediction Model")
    st.info("An advanced stacking ensemble is used to improve prediction accuracy. "
            "The trained model is stored, and selecting a custom input dynamically updates predictions.")

    # Let user select the target variable for prediction
    target_col = st.selectbox("Select Target Variable for Prediction", df.columns, key="target_select")

    if target_col:
        # Separate features and target
        X = df.drop(target_col, axis=1)
        y = df[target_col]

        # Train the model only if not already stored or if target changes
        if ("trained_model" not in st.session_state) or (st.session_state.get("target_col") != target_col):
            st.write("Training advanced stacking ensemble model...")
            from sklearn.model_selection import train_test_split, GridSearchCV

            if y.dtype in ['int64', 'float64']:
                st.write("Detected Regression Problem")
                from sklearn.ensemble import StackingRegressor, RandomForestRegressor, GradientBoostingRegressor
                from sklearn.linear_model import Ridge
                from sklearn.metrics import mean_squared_error, r2_score

                estimators = [
                    ('rf', RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)),
                    ('gbr', GradientBoostingRegressor(n_estimators=100, random_state=42))
                ]
                model = StackingRegressor(
                    estimators=estimators,
                    final_estimator=Ridge(),
                    n_jobs=-1
                )
                param_grid = {
                    'rf__n_estimators': [100, 200],
                    'gbr__n_estimators': [100, 200],
                }
                grid = GridSearchCV(model, param_grid, cv=5, n_jobs=-1)
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                grid.fit(X_train, y_train)
                best_model = grid.best_estimator_
                y_pred = best_model.predict(X_test)
                mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                st.session_state["trained_model"] = best_model
                st.session_state["model_metrics"] = {"MSE": mse, "R2": r2}
                st.session_state["problem_type"] = "regression"
            else:
                st.write("Detected Classification Problem")
                from sklearn.ensemble import StackingClassifier, RandomForestClassifier, GradientBoostingClassifier
                from sklearn.linear_model import LogisticRegression
                from sklearn.metrics import accuracy_score, classification_report

                estimators = [
                    ('rf', RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)),
                    ('gbc', GradientBoostingClassifier(n_estimators=100, random_state=42))
                ]
                model = StackingClassifier(
                    estimators=estimators,
                    final_estimator=LogisticRegression(),
                    n_jobs=-1
                )
                param_grid = {
                    'rf__n_estimators': [100, 200],
                    'gbc__n_estimators': [100, 200],
                }
                grid = GridSearchCV(model, param_grid, cv=5, n_jobs=-1)
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                grid.fit(X_train, y_train)
                best_model = grid.best_estimator_
                y_pred = best_model.predict(X_test)
                acc = accuracy_score(y_test, y_pred)
                st.session_state["trained_model"] = best_model
                st.session_state["model_metrics"] = {"Accuracy": acc}
                st.session_state["problem_type"] = "classification"
            st.session_state["target_col"] = target_col
        else:
            best_model = st.session_state["trained_model"]
            st.write("Loaded trained model from session state.")
            st.write("Model Metrics:", st.session_state["model_metrics"])

        # -------------------------------
        # Dynamic Prediction with Custom Input for a Single Feature
        # -------------------------------
        st.write("### Make a Prediction with Custom Input")
        st.info("Select a feature and modify its value. The prediction updates dynamically.")

        # Dropdown for selecting which feature to override
        selected_feature = st.selectbox("Select Feature to Modify", options=list(X.columns), key="custom_feature_select")

        # Dictionary to store session state input values
        if "input_values" not in st.session_state:
            st.session_state["input_values"] = {col: X[col].median() if X[col].dtype in ['int64', 'float64'] else X[col].mode()[0] for col in X.columns}

        # Provide a number input for numeric features or dropdown for categorical
        if selected_feature:
            if X[selected_feature].dtype in ['int64', 'float64']:
                custom_value = st.number_input(
                    f"Enter custom value for '{selected_feature}'",
                    value=st.session_state["input_values"].get(selected_feature, X[selected_feature].median()),
                    key=f"input_{selected_feature}",
                    on_change=lambda: st.session_state["input_values"].update({selected_feature: st.session_state[f"input_{selected_feature}"]})
                )
            else:
                unique_vals = sorted(X[selected_feature].unique())
                custom_value = st.selectbox(
                    f"Select custom value for '{selected_feature}'",
                    options=unique_vals,
                    index=unique_vals.index(st.session_state["input_values"].get(selected_feature, X[selected_feature].mode()[0])),
                    key=f"input_{selected_feature}",
                    on_change=lambda: st.session_state["input_values"].update({selected_feature: st.session_state[f"input_{selected_feature}"]})
                )

        # Generate the prediction based on the updated session state
        input_vector = pd.DataFrame([st.session_state["input_values"]])

        # Perform prediction dynamically when user changes values
        prediction = best_model.predict(input_vector)
        st.write("### Prediction Result:", prediction[0])

