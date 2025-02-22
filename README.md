# Diversion-Hackathon
<!DOCTYPE html>
<html>
    <head>
        <title>Ez-Viz</title>
        <meta charset="utf-8">
        <meta name = "viewport" content="width=device-width, initial-scale=1.0">
        <link rel="stylesheet" href="styles.css">
    </head>
    <body>
        <h1>Importing necessary models :</h1>
        <img src="models.jpg" alt="models required for Ez-Viz">
        <h3>
            <ul>
                <li><i>RandomForestClassifier</i> : An ensemble learning method using multiple decision trees to improve accuracy and reduce overfitting. Ideal for classification tasks.</li>
                <li><i>GradientBoostingClassifier</i> : Another ensemble technique that builds models sequentially, optimizing errors from previous models. Suitable for complex datasets.</li>
                <li><i>AdaBoostClassifier</i> : An adaptive boosting method that combines weak classifiers iteratively to form a strong classifier. Effective for reducing bias and variance.</li>
                <li><i>LinearRegression</i> : A basic model for predicting continuous values by fitting a linear relationship between features and targets.</li>
                <li><i>LogisticRegression</i> : A statistical model for binary or multi-class classification problems using a sigmoid function for prediction.</li>
                <li><i>DecisionTreeClassifier</i> : A tree-based model that splits data into branches based on feature conditions for classification.</li>
                <li><i>SVC (Support Vector Classifier)</i> : A model that finds the optimal hyperplane to separate data into classes, effective for high-dimensional spaces.</li>
                <li><i>KNeighborsClassifier</i> : A simple algorithm that classifies a sample based on the majority class among its 'k' nearest neighbors.</li>
            </ul>
        </h3>
        <h1>Seperating numericals and categorical values :</h1>
        <img src="sep nem and cat val.jpg" alt="image of the code for the same">
        <p>The python code shown in the above image has the following functionality:</p>
        <ul>
            <li>
            <strong>Read and Preview Data</strong>: Reads the uploaded CSV file into a 
            Pandas DataFrame (<span class="highlight">df</span>) and displays its 
            first few rows in Streamlit.
            </li>
            <li>
            <strong>Feature Separation</strong>:
            <ul>
                <li>
                <span class="highlight">num_features</span> identifies columns that 
                contain numerical data (using 
                <span class="highlight">include=[np.number]</span>).
                </li>
                <li>
                <span class="highlight">cat_features</span> identifies columns that 
                contain non-numerical (categorical) data (using 
                <span class="highlight">exclude=[np.number]</span>).
                </li>
            </ul>
            </li>
            <li>
            <strong>Display Column Analysis</strong>: Prints out the numerical and 
            categorical features using Streamlit’s 
            <span class="highlight">st.write()</span>.
            </li>
        </ul>
</body>

</html>