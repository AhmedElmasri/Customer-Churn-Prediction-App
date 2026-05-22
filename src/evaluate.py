"""Evaluation helpers for classification models."""

from __future__ import annotations

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def evaluate_classifier(model, X_train, y_train, X_test, y_test) -> dict:
    """Evaluate one fitted classifier on train and test sets.

    Note: the original notebook printed `classification_report` with arguments
    reversed in some cells. This version uses the correct order:
    classification_report(y_true, y_pred).
    """
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    return {
        "train_accuracy": accuracy_score(y_train, y_train_pred),
        "test_accuracy": accuracy_score(y_test, y_test_pred),
        "train_precision_macro": precision_score(y_train, y_train_pred, average="macro", zero_division=0),
        "test_precision_macro": precision_score(y_test, y_test_pred, average="macro", zero_division=0),
        "train_recall_macro": recall_score(y_train, y_train_pred, average="macro", zero_division=0),
        "test_recall_macro": recall_score(y_test, y_test_pred, average="macro", zero_division=0),
        "train_f1_macro": f1_score(y_train, y_train_pred, average="macro", zero_division=0),
        "test_f1_macro": f1_score(y_test, y_test_pred, average="macro", zero_division=0),
        "test_confusion_matrix": confusion_matrix(y_test, y_test_pred).tolist(),
        "train_report": classification_report(
            y_train,
            y_train_pred,
            target_names=["No", "Yes"],
            digits=4,
            zero_division=0,
        ),
        "test_report": classification_report(
            y_test,
            y_test_pred,
            target_names=["No", "Yes"],
            digits=4,
            zero_division=0,
        ),
    }
