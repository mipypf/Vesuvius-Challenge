from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import confusion_matrix


def fbeta_numpy(targets: np.ndarray, preds: np.ndarray, beta: float = 0.5, smooth: float = 1e-5):
    """
    https://www.kaggle.com/competitions/vesuvius-challenge-ink-detection/discussion/397288
    """
    assert 0 <= beta <= 1

    y_true_count = targets.sum()
    ctp = preds[targets == 1].sum()
    cfp = preds[targets == 0].sum()
    beta_squared = beta * beta

    c_precision = ctp / (ctp + cfp + smooth)
    c_recall = ctp / (y_true_count + smooth)
    dice = (1 + beta_squared) * (c_precision * c_recall) / (beta_squared * c_precision + c_recall + smooth)

    return dice, c_precision, c_recall


def calc_fbeta(
    labels: np.ndarray,
    labels_pred: np.ndarray,
    masks: Optional[np.ndarray] = None,
    beta: float = 0.5,
    show_confusion_matrix: bool = False,
):
    assert labels.shape == labels_pred.shape

    labels = labels.astype(np.int32)
    if masks is None:
        masked_labels = labels.flatten()
        masked_labels_pred = labels_pred.flatten()
    else:
        assert labels.shape == masks.shape
        assert masks.dtype == bool
        masked_labels = labels[masks]
        masked_labels_pred = labels_pred[masks]

    ret = list()
    for threshold in np.arange(0.1, 0.91, 0.05):
        threshold = np.round(threshold, 3)
        fscore, precision, recall = fbeta_numpy(masked_labels, masked_labels_pred >= threshold, beta=beta)
        logger.info(f"threshold: {threshold:.2f}, fscore={fscore:.3f} precision={precision:.3f} recall={recall:.3f}")
        ret.append(dict(threshold=threshold, fscore=fscore, precision=precision, recall=recall))

    df = pd.DataFrame(ret)
    df_best = df.sort_values("fscore", ascending=False).iloc[0]

    logger.info(
        f"best_th: {df_best.threshold:.2f}, fscore={df_best.fscore:.3f}, "
        + f"precision={df_best.precision:.3f}, recall={df_best.recall:.3f}"
    )

    if show_confusion_matrix:
        cm = confusion_matrix(labels, labels_pred > df_best.threshold)
        cm = pd.DataFrame(
            cm,
            index=["rot0-180 (GT)", "rot90-270 (GT)"],
            columns=["rot0-180 (Pred)", "rot90-270 (Pred)"],
        )
        logger.info(f"Confusion Matrix\n{cm.to_markdown(mode='str')}")

    return df_best.to_dict(), ret
