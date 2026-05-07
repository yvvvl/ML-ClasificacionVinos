from src.carga_data import cargar_datos_vino, explorar_datos, separar_variables
from src.evaluacion import  evaluar_modelo
from src.visualizacion import visualizar_arbol
from src.modelo import entrenar_arbol_decision

if __name__ == "__main__":
    
    df= cargar_datos_vino()

    print("Datos cargados con éxito.")
    explorar_datos(df)


    X, y = separar_variables(df)

    print("Variables separadas con éxito.")
    print(' ')
    print("Primeras filas de las features (X):")
    print(X.head())
    print(' ')
    print("Primeras filas del target (y):")
    print(' ')
    print(y.head())

    #ahora ya tenemos las variables separadas en X (features) e y (target) para poder entrenar nuestro modelo de machine learning.

    modelo, X_test, y_test = entrenar_arbol_decision(X, y)

    print("Modelo entrenado con éxito.")

    evaluar_modelo(modelo, X_test, y_test)

    print("Modelo evaluado con éxito.")
    feature_names = X.columns
    class_names = [str(cls) for cls in sorted(y.unique())]
    visualizar_arbol(modelo, feature_names, class_names)