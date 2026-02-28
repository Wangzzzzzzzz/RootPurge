from sklearn.metrics import confusion_matrix

def calculate_fnr_fpr(y_pred, y_true):
    """
    Calculate False Alarm Rate and Escape Rate.

    This function computes the following metrics based on the provided
    true labels (y_true) and predicted labels (y_pred):
    - False Alarm Rate: The proportion of true positive class samples (0)
      that are incorrectly classified as negative class (1).
    - Escape Rate: The proportion of true negative class samples (1)
      that are incorrectly classified as positive class (0).

    Parameters:
    y_true (list or array-like): True labels, expected to be a list of binary labels (e.g., [0, 1, 0, 1]).
    y_pred (list or array-like): Predicted labels, with the same format as y_true.

    Returns:
    tuple: A tuple containing two float values (false_alarm_rate, escape_rate)
        - false_alarm_rate (float): False Alarm Rate
        - escape_rate (float): Escape Rate
    """

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    false_alarm_rate = fp / (fp + tn)
    escape_rate = fn / (fn + tp)
    return false_alarm_rate, escape_rate
