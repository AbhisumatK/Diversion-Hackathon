import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import LabelEncoder, StandardScaler

st.title("Machine Learning Visualization Tool")

# Step 1: Dataset Upload
st.sidebar.header("Upload Your CSV File")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### Dataset Preview")
    st.write(df.head())

    # Step 2: Data Cleaning
    df.drop_duplicates(inplace=True)
    df.fillna(df.median(numeric_only=True), inplace=True)
    for col in df.select_dtypes(include=['object']).columns:
        df[col].fillna(df[col].mode()[0], inplace=True)

    # Step 3: Column Analysis
    st.write("### Column Analysis")
    numerical_cols = df.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=['number']).columns.tolist()
    st.write("Numerical Columns:", numerical_cols)
    st.write("Categorical Columns:", categorical_cols)

    # Step 4: Description of Data
    st.write("### Summary Statistics")
    st.write(df[numerical_cols].describe())

    # Encode categorical variables
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    # Data Filtering
    a = numerical_cols[1]
    b = numerical_cols[2]
    st.write("### Data Filtering")
    filter_conditions = st.text_area(f"Enter filter conditions (`e.g., {a} > {df[a].values[1]} & {b} < {df[b].values[3]}`)")
    if filter_conditions:
        try:
            df = df.query(filter_conditions)
            st.write("Filtered Dataset Preview")
            st.write(df.head())
        except Exception as e:
            st.error(f"Invalid filter condition: {e}")

    # Normalization / Scaling
    st.write("### Normalize/Scale Numerical Values")
    if st.checkbox("Normalize/Scale Numerical Columns"):
        scaler = StandardScaler()
        df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
        st.write("Normalized Dataset Preview")
        st.write(df.head())

    # Step 5: Data Visualizations
    st.write("### Data Visualizations")

    # Graph Suggestion
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