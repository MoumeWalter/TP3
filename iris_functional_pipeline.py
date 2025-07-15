"""
iris_functional_pipeline.py

Pipeline de traitement fonctionnel du dataset Iris
Respect des principes de la programmation fonctionnelle :
- Fonctions pures
- Immutabilité
- Abstractions fonctionnelles : map, filter, reduce
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from functools import reduce
from typing import Callable, List


# -------------------------------
# 1. Chargement des données (fonction pure)
# -------------------------------
def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


# -------------------------------
# 2. Nettoyage des données
# -------------------------------
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    return df.dropna()


# -------------------------------
# 3. Mapping des features numériques (immuable)
# -------------------------------
def extract_numeric_features(df: pd.DataFrame) -> pd.DataFrame:
    return df.select_dtypes(include=['float64'])


# -------------------------------
# 4. Agrégation fonctionnelle : moyenne des colonnes
# -------------------------------
def average_column(col: pd.Series) -> float:
    return col.mean()


def average_all_columns(df: pd.DataFrame) -> dict:
    return dict(map(lambda col: (col, average_column(df[col])), df.columns))


# -------------------------------
# 5. Visualisation fonctionnelle
# -------------------------------
def plot_correlation_heatmap(df: pd.DataFrame, save_path="correlation_heatmap.png") -> None:
    corr = df.corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm')
    plt.title("Matrice de corrélation")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.clf()


def plot_scatter(df: pd.DataFrame, save_path="scatter_plot.png") -> None:
    sns.pairplot(df)
    plt.suptitle("Scatter matrix des features", y=1.02)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.clf()


# -------------------------------
# 6. Clustering (k-means)
# -------------------------------
def perform_clustering(df: pd.DataFrame, n_clusters: int = 3) -> List[int]:
    model = KMeans(n_clusters=n_clusters, random_state=42)
    model.fit(df)
    return model.labels_


# -------------------------------
# 7. Pipeline principal
# -------------------------------
def iris_pipeline(csv_path: str) -> None:
    # Étapes fonctionnelles chaînées
    df = load_data(csv_path)
    df = clean_data(df)
    df_numeric = extract_numeric_features(df)

    # Statistiques
    averages = average_all_columns(df_numeric)
    print("Moyennes des colonnes :")
    for col, val in averages.items():
        print(f"{col} : {val:.2f}")

    # Visualisations
    plot_correlation_heatmap(df_numeric)
    plot_scatter(df_numeric)

    # Clustering
    labels = perform_clustering(df_numeric)
    df['Cluster'] = labels

    # Sauvegarde du résultat
    df.to_csv("iris_with_clusters.csv", index=False)
    print("Pipeline terminé. Résultats enregistrés dans 'iris_with_clusters.csv'")


# -------------------------------
# Exécution
# -------------------------------
if __name__ == "__main__":
    iris_pipeline("Iris.csv")
