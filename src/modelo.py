from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from carga_data import X, y

def entrenar_arbol_decision(X, y):
    """
    Entrena un modelo de Árbol de Decisión y evalúa su rendimiento.
    """
    # 1. Dividir los datos en conjunto de entrenamiento (80%) y prueba (20%)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 2. Inicializar el clasificador de Árbol de Decisión
    # random_state asegura que los resultados sean reproducibles
    modelo = DecisionTreeClassifier(random_state=42)
    
    # 3. Entrenar el modelo con los datos de entrenamiento
    modelo.fit(X_train, y_train)
    
    # 4. Realizar predicciones sobre los datos de prueba
    y_pred = modelo.predict(X_test)
    
    # 5. Evaluar el modelo
    precision = accuracy_score(y_test, y_pred)
    print(f"Precisión del modelo: {precision * 100:.2f}%\n")
    
    print("Reporte de Clasificación:")
    print(classification_report(y_test, y_pred))
    
    return modelo

if __name__ == "__main__":
    print("Entrenando Árbol de Decisión para la clasificación de vinos...\n")
    modelo_entrenado = entrenar_arbol_decision(X, y)
