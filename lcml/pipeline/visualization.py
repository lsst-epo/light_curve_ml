import itertools
try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None
import numpy as np

from lcml.utils.basic_logging import BasicLogging


logger = BasicLogging.getLogger(__name__)


def normalizeConfusionMatrix(matrix):
    return matrix.astype("float") / matrix.sum(axis=1)[:, np.newaxis]


def plotConfusionMatrix(mat, classes, normalize=False, title="Confusion matrix",
                        cmap=None):
    """Plots a confusion matrix and its classes

    :param mat: ndarray confusion matrix
    :param classes: list of class names
    :param normalize: if True, normalize matrix cells
    :param title: figure title
    :param cmap: color map for intensity scale
    """
    if not plt:
        logger.warning("Skipping plot since Matplotlib unavailable")
        return

    if normalize:
        mat = normalizeConfusionMatrix(mat)

    cmap = cmap if cmap else plt.cm.Blues
    plt.figure()
    plt.imshow(mat, interpolation="nearest", cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = ".2f" if normalize else "d"
    thresh = mat.max() / 2.
    for i, j in itertools.product(range(mat.shape[0]), range(mat.shape[1])):
        plt.text(j, i, format(mat[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if mat[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.show()
    return mat
