# Models
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder

def select_model(type: int):
    match type:
        case 1:
            return LogisticRegression()
        case 2:
            return LinearRegression()
        case 3:
            return RandomForestClassifier()
        case 4:
            return GradientBoostingClassifier()
        case 5:
            return SVC()
        case 6:
            return KNeighborsClassifier()