import itertools
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties


import numpy as np

from lcml.utils.basic_logging import BasicLogging


logger = BasicLogging.getLogger(__name__)


def normalizeConfusionMatrix(matrix):
    return matrix.astype("float") / matrix.sum(axis=1)[:, np.newaxis]


def plotConfusionMatrix(matrix, classes, savePath=None, normalize=True,
                        title="Confusion matrix", cmap=None):
    """Plots a confusion matrix and its classes.

    :param matrix: ndarray confusion matrix
    :param classes: list of class names
    :param savePath: Full path where plot will be saved
    :param normalize: if True, normalize matrix cells
    :param title: figure title
    :param cmap: color map for intensity scale
    """
    if normalize:
        matrix = normalizeConfusionMatrix(matrix)

    cmap = cmap if cmap else plt.cm.Blues
    plt.figure(figsize=(8, 6))
    plt.imshow(matrix, interpolation="nearest", cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = ".2f" if normalize else "d"
    thresh = matrix.max() / 2.
    for i, j in itertools.product(range(matrix.shape[0]),
                                  range(matrix.shape[1])):
        cellValue = format(matrix[i, j], fmt) if matrix[i, j] else ""
        color = "white" if matrix[i, j] > thresh else "black"
        plt.text(j, i, cellValue, horizontalalignment="center", color=color)

    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    if savePath:
        logger.info("Saving plot: '%s' to: %s", title, savePath)
        plt.savefig(savePath, bbox_inches="tight")
    else:
        plt.show()
    return matrix


def contourPlot(x, y, z, savePath=None, title="Contour Plot", xLabel=None,
                yLabel=None):
    cs = plt.contourf(x, y, z, corner_mask=True)

    fontP = FontProperties()
    fontP.set_size("small")

    nm, lbl = cs.legend_elements()
    plt.legend(nm, lbl, title=None, prop=fontP, loc="center left",
               bbox_to_anchor=(1, 0.5))
    plt.contour(cs, colors="k")
    plt.title(title)
    if xLabel:
        plt.xlabel(xLabel)
    if yLabel:
        plt.ylabel(yLabel)

    # Plot grid
    plt.grid(c="k", ls="-", alpha=0.3)
    if savePath:
        logger.info("Saving contour plot to %s", savePath)
        plt.savefig(savePath, bbox_inches="tight")
    else:
        plt.show()
