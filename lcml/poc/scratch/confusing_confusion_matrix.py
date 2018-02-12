import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    y_actu = pd.Series([2, 0, 2, 2, 0, 1, 1, 2, 2, 0, 1, 2], name='Actual')
    y_pred = pd.Series([0, 0, 2, 1, 0, 2, 1, 0, 2, 0, 2, 2], name='Predicted')
    df_confusion = pd.crosstab(y_actu, y_pred, margins=True)
    print(df_confusion)

    noMarginConfusion = pd.crosstab(y_actu, y_pred, margins=False)
    plot_confusion_matrix(noMarginConfusion)


def plot_confusion_matrix(df_confusion, title='Confusion matrix',
                          cmap=plt.cm.gray_r):
    plt.matshow(df_confusion, cmap=cmap)  # imshow
    # plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(df_confusion.columns))
    plt.xticks(tick_marks, df_confusion.columns, rotation=45)
    plt.yticks(tick_marks, df_confusion.index)
    # plt.tight_layout()
    plt.ylabel(df_confusion.index.name)
    plt.xlabel(df_confusion.columns.name)
    plt.show()



if __name__ == "__main__":
    main()
