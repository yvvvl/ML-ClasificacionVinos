from sklearn import tree
import matplotlib.pyplot as plt


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


