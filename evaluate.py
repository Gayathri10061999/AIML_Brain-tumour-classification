from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

def evaluate_model(model, val_gen):
    preds = model.predict(val_gen)
    y_pred = np.argmax(preds, axis=1)
    y_true = val_gen.classes

    print(classification_report(y_true, y_pred))
    print(confusion_matrix(y_true, y_pred))
