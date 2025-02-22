# Models
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder

def select_model(type: str):
    if type == "Logistic Regression":
        model = LogisticRegression()
    if type == "Linear Regression":
        model = LinearRegression()
    if type == "Random Forest":
        model = RandomForestClassifier()
    if type == "Gradient Boosting":
        model = GradientBoostingClassifier()
    if type == "SVC":
        model = SVC()
    if type == "KNN":
        model = KNeighborsClassifier()
    return model