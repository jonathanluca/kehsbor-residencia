"""Métricas macro e a validação em que o site não passa de um lado para o outro."""

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedGroupKFold

from classificacao.modelos import criar_modelos
from classificacao.representacao import representar, sinais_da_manchete

COLUNAS = ["acurácia", "precisão", "recall", "f1"]


def medir(y_verdadeiro, y_predito):
    """Média das duas classes.

    Numa dobra com quase só falso, a acurácia sobe ao chutar a maioria.
    O F1 cai quando o modelo erra a classe menor. É o número para comparar.
    """
    return {
        "acurácia": accuracy_score(y_verdadeiro, y_predito),
        "precisão": precision_score(y_verdadeiro, y_predito, average="macro", zero_division=0),
        "recall": recall_score(y_verdadeiro, y_predito, average="macro", zero_division=0),
        "f1": f1_score(y_verdadeiro, y_predito, average="macro", zero_division=0),
    }


def treinar_e_prever(X_treino, y_treino, X_teste):
    """Ajusta cada modelo do zero e devolve a previsão do teste."""
    previsoes = {}
    for nome, modelo in criar_modelos().items():
        modelo.fit(X_treino, y_treino)
        previsoes[nome] = modelo.predict(X_teste)
    return previsoes


def tabela(previsoes, y_teste):
    """Uma linha por modelo, com as quatro métricas."""
    linhas = [{"modelo": nome, **medir(y_teste, y_pred)} for nome, y_pred in previsoes.items()]
    return pd.DataFrame(linhas).set_index("modelo")


def desenhar_matrizes(previsoes, y_teste):
    """Matriz de confusão de cada modelo. O rótulo 0 é falso e o 1 é verdadeiro."""
    fig, eixos = plt.subplots(1, len(previsoes), figsize=(14, 4))
    for eixo, (nome, y_pred) in zip(eixos, previsoes.items()):
        ConfusionMatrixDisplay.from_predictions(
            y_teste,
            y_pred,
            display_labels=["falso", "verdadeiro"],
            cmap="Blues",
            colorbar=False,
            ax=eixo,
        )
        eixo.set_title(nome)
    fig.tight_layout()
    return fig


def validar_por_dominio(base, n_splits=4, random_state=42):
    """Quatro dobras em que o mesmo site fica só de um lado.

    Um GroupKFold comum isolava boatos.org, só falso, e as linhas sem URL,
    só verdadeiro. A dobra estratificada força as duas classes em cada teste.
    O TF-IDF é reajustado dentro da dobra, senão o teste entraria no vocabulário.
    """
    dobras = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    registro = []
    grupos = base["dominio"]
    for numero, (idx_treino, idx_teste) in enumerate(
        dobras.split(base, base["rotulo"], grupos), start=1
    ):
        treino = base.iloc[idx_treino]
        teste = base.iloc[idx_teste]
        X_treino, X_teste = representar(
            treino["titulo"],
            sinais_da_manchete(treino),
            teste["titulo"],
            sinais_da_manchete(teste),
        )
        sites = ", ".join(sorted(teste["dominio"].unique()))
        print(f"Dobra {numero}: teste {len(teste)} | {sites}")
        previsoes = treinar_e_prever(X_treino, treino["rotulo"], X_teste)
        for nome, y_pred in previsoes.items():
            registro.append({"modelo": nome, "dobra": numero, **medir(teste["rotulo"], y_pred)})
    return pd.DataFrame(registro)


def resumo(por_dobra):
    """Média das dobras. Este é o número para dizer qual modelo segurou."""
    return por_dobra.groupby("modelo")[COLUNAS].mean().round(3)
