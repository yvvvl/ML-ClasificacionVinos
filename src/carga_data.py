from sklearn.datasets import load_wine
import pandas as pd
import numpy as np
import pandas as pd


#convertimos ScikitLearn en DataFrame
def cargar_datos_vino():
    """
    Carga el dataset de vinos y lo convierte en DataFrame de Pandas.
    """
    wine_data = load_wine(as_frame=True)
    df = wine_data.frame
    return df

#cargamos los datos
df = cargar_datos_vino()


def explorar_datos(df):
    """
    Explora el DataFrame de Pandas.
    """
    print(len(df))
    print(df['target'].unique())
    print(df.describe())
    print(df.columns)
    print(df.head())

#exploramos los datos
explorar_datos(df)

