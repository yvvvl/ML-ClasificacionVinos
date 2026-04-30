from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

def entrenar_arbol_decision(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    modelo = DecisionTreeClassifier(random_state=42)
    modelo.fit(X_train, y_train)
    return modelo, X_test, y_test


