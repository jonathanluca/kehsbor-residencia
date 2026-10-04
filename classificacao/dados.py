"""Leitura das manchetes e o corte por site.

O modelo classifica o título. URL e data ficam de fora das colunas, mas o
texto ainda carrega o jeito de escrever de cada site. O teste guarda sites
inteiros para essa pista não aparecer dos dois lados.
"""

from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
ARQUIVO = RAIZ / "dados" / "dataset_recogna_politica_balanceado.csv"
# Corte único. A comparação quando o site muda de verdade é a média das
# dobras em avaliacao.validar_por_dominio: este trio mistura pedaços delas.
SITES_TESTE = frozenset(
    {
        "projetocomprova.com.br",
        "e-farsas.com",
        "g1.globo.com",
    }
)


def carregar_manchetes(caminho=ARQUIVO):
    """Abre o CSV e fica só com manchetes que têm um rótulo.

    Título vazio não tem o que classificar. A mesma frase com os dois
    rótulos iria para o treino e para o teste ao mesmo tempo.
    """
    base = pd.read_csv(caminho)
    base = base.dropna(subset=["titulo"]).copy()
    base["titulo"] = base["titulo"].astype(str).str.strip()
    base = base.loc[base["titulo"].ne("")].copy()
    um_rotulo = base.groupby("titulo")["rotulo"].transform("nunique").eq(1)
    base = base.loc[um_rotulo].reset_index(drop=True)
    base["dominio"] = base["url"].fillna("sem url")
    return base


def separar_por_sites(base, sites_teste=SITES_TESTE):
    """Treino e teste sem o mesmo site dos dois lados.

    Sortear linhas deixa o estilo do site no treino e no teste e infla a
    acurácia. Aqui o teste é projetocomprova, e-farsas e g1.
    """
    no_teste = base["dominio"].isin(sites_teste)
    return base.loc[~no_teste].copy(), base.loc[no_teste].copy()
