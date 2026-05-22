"""Model definitions taken from the original notebook."""

from __future__ import annotations

from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from src.config import RANDOM_STATE


def get_models() -> dict:
    """Return the classifiers configured as in the original notebook."""
    models = {
        "KNN": KNeighborsClassifier(n_neighbors=13, p=2),
        "Naive Bayes": GaussianNB(),
        "SVM": SVC(C=1, kernel="rbf"),
        "Decision Tree": DecisionTreeClassifier(
            criterion="gini",
            splitter="best",
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=0,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=30,
            criterion="gini",
            max_depth=None,
            min_samples_split=8,
            min_samples_leaf=2,
            random_state=0,
        ),
        "AdaBoost": AdaBoostClassifier(
            n_estimators=50,
            learning_rate=1.0,
            random_state=0,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            learning_rate=0.1,
            n_estimators=100,
            max_depth=3,
            random_state=0,
        ),
    }

    # XGBoost was used in the notebook. Keep it optional so the rest of the
    # pipeline still works if xgboost is not installed.
    try:
        import xgboost as xgb

        models["XGBoost"] = xgb.XGBClassifier(
            learning_rate=0.1,
            max_depth=3,
            n_estimators=100,
            random_state=0,
            eval_metric="logloss",
            tree_method="hist",
            n_jobs=1,
            verbosity=0,
        )
    except ImportError:
        pass

    return models
