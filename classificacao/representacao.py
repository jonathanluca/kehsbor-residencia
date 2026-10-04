from functools import lru_cache

import numpy as np
import nltk
import pandas as pd
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler


@lru_cache(maxsize=1)
def palavras_de_parada():
    nltk.download("stopwords", quiet=True)
    return stopwords.words("portuguese")


def sinais_da_manchete(tabela):
    titulo = tabela["titulo"]
    return pd.DataFrame(
        {
            "tem_aspas": titulo.str.contains(r'["“”\']', regex=True).astype(float),
            "tem_dois_pontos": titulo.str.contains(":", regex=False).astype(float),
            "comeca_minuscula": titulo.str.match(r"^[a-záàâãéêíóôõúüç]").astype(float),
            "tem_numero": titulo.str.contains(r"\d", regex=True).astype(float),
            "prob_negativo": tabela["titulo_prob_negativo"].astype(float),
            "prob_positivo": tabela["titulo_prob_positivo"].astype(float),
        },
        index=tabela.index,
    )


def representar(titulos_treino, sinais_treino, titulos_teste, sinais_teste):
    # Palavras e escala saem só do treino, para o teste não vazar.
    vetor = TfidfVectorizer(
        lowercase=True,
        stop_words=palavras_de_parada(),
        ngram_range=(1, 2),
        min_df=2,
        max_features=5000,
    )
    escala = StandardScaler()
    treino_tf = vetor.fit_transform(titulos_treino).toarray()
    teste_tf = vetor.transform(titulos_teste).toarray()
    treino_num = escala.fit_transform(sinais_treino)
    teste_num = escala.transform(sinais_teste)
    return np.hstack([treino_tf, treino_num]), np.hstack([teste_tf, teste_num])
