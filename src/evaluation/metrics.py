import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


def _predict_labels(model, ds):
    y_true = []
    y_pred = []
    for x, y in ds:
        probs = model.predict(x, verbose=0)
        y_true.extend(y.numpy().tolist())
        y_pred.extend(np.argmax(probs, axis=1).tolist())
    return np.array(y_true), np.array(y_pred)


def compute_metrics(model, ds, class_names):
    """
    Retorna un diccionario serializable (para metrics.json).
    """
    y_true, y_pred = _predict_labels(model, ds)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": classification_report(
            y_true, y_pred, target_names=class_names, output_dict=True
        ),
    }