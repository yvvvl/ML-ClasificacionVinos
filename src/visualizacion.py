import os
import sys

# Asegura que el proyecto raíz esté en el path cuando se ejecuta como script.
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import tree
import matplotlib.pyplot as plt

from src.carga_data import cargar_datos_vino, separar_variables


def entrenar_arbol_decision(X, y):
    """Entrena un modelo de árbol de decisión con los datos dados."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    modelo = DecisionTreeClassifier(random_state=42)
    modelo.fit(X_train, y_train)
    return modelo, X_test, y_test


def visualizar_arbol(modelo, feature_names, class_names):
    """Dibuja el árbol de decisión usando Matplotlib."""
    plt.figure(figsize=(20, 12))
    tree.plot_tree(
        modelo,
        feature_names=feature_names,
        class_names=class_names,
        filled=True,
        rounded=True,
        impurity=False,
        fontsize=10,
    )
    plt.title("Árbol de decisión entrenado")
    plt.tight_layout()
    plt.savefig("arbol_decision.png", dpi=200)
    plt.show()


def main():
    df = cargar_datos_vino()
    X, y = separar_variables(df)
    modelo, X_test, y_test = entrenar_arbol_decision(X, y)
    class_names = [str(c) for c in sorted(y.unique())]
    visualizar_arbol(modelo, feature_names=X.columns, class_names=class_names)


if __name__ == "__main__":
    main()


