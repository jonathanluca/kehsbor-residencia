from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC


def criar_modelos():
    return {
        "Regressão logística": LogisticRegression(max_iter=1000, random_state=42),
        "SVM": LinearSVC(random_state=42, dual="auto"),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            n_jobs=-1,
        ),
    }
