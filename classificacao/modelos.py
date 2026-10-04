"""Três classificadores na mesma matriz, para a comparação ser justa."""

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC


def criar_modelos():
    """Instâncias novas a cada chamada.

    Reusar um modelo já treinado misturaria a dobra anterior. Sem
    class_weight: o corte único está equilibrado, e o desvio da dobra
    com quase só boato aparece no F1.
    """
    return {
        "Regressão logística": LogisticRegression(max_iter=1000, random_state=42),
        "SVM": LinearSVC(random_state=42, dual="auto"),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            n_jobs=-1,
        ),
    }
