from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
ARQUIVO = RAIZ / "dados" / "dataset_recogna_politica_balanceado.csv"
SITES_TESTE = frozenset(
    {
        "projetocomprova.com.br",
        "e-farsas.com",
        "g1.globo.com",
    }
)


def carregar_manchetes(caminho=ARQUIVO):
    base = pd.read_csv(caminho)
    base = base.dropna(subset=["titulo"]).copy()
    base["titulo"] = base["titulo"].astype(str).str.strip()
    base = base.loc[base["titulo"].ne("")].copy()
    um_rotulo = base.groupby("titulo")["rotulo"].transform("nunique").eq(1)
    base = base.loc[um_rotulo].reset_index(drop=True)
    base["dominio"] = base["url"].fillna("sem url")
    return base


def separar_por_sites(base, sites_teste=SITES_TESTE):
    no_teste = base["dominio"].isin(sites_teste)
    return base.loc[~no_teste].copy(), base.loc[no_teste].copy()
