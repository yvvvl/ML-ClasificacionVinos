from sklearn.metrics import accuracy_score, classification_report

def evaluar_modelo(modelo, X_test, y_test):
    y_pred = modelo.predict(X_test)
    precision = accuracy_score(y_test, y_pred)
    print(f"Precisión del modelo: {precision * 100:.2f}%\n")
    print("Reporte de Clasificación:")
    print(classification_report(y_test, y_pred))

    return precision