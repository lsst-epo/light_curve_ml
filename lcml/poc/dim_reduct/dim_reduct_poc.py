#!/usr/bin/env python3
import time

import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler

from lcml.pipeline.database.sqlite_db import selectFeaturesLabels
from lcml.utils.basic_logging import BasicLogging
from lcml.utils.context_util import jsonConfig


BasicLogging.initLogging()
logger = BasicLogging.getLogger(__name__)


def testIris():
    """sklearn library example"""
    iris = datasets.load_iris()

    X = iris.data
    y = iris.target
    target_names = iris.target_names

    components = 2

    X_std = StandardScaler().fit_transform(X)

    # PCA transforms features into a new space
    pca = PCA(n_components=components)
    X_r = pca.fit(X_std).transform(X_std)


    lda = LinearDiscriminantAnalysis(n_components=components)
    X_r2 = lda.fit(X_std, y).transform(X_std)

    # Percentage of variance explained for each components
    print('explained variance ratio (first two components): %s'
          % str(pca.explained_variance_ratio_))

    plt.figure()
    colors = ['navy', 'turquoise', 'darkorange']
    lw = 2

    for color, i, target_name in zip(colors, [0, 1, 2], target_names):
        plt.scatter(X_r[y == i, 0], X_r[y == i, 1], color=color, alpha=.8, lw=lw,
                    label=target_name)
    plt.legend(loc='best', shadow=False, scatterpoints=1)
    plt.title('PCA of IRIS dataset')

    plt.figure()
    for color, i, target_name in zip(colors, [0, 1, 2], target_names):
        plt.scatter(X_r2[y == i, 0], X_r2[y == i, 1], alpha=.8, color=color,
                    label=target_name)
    plt.legend(loc='best', shadow=False, scatterpoints=1)
    plt.title('LDA of IRIS dataset')

    plt.show()


def main():
    conf = jsonConfig("pipeline.json")
    dbConf = conf["database"]
    dbConf["dbPath"] = "data/macho/macho_processed.db"

    X, y = selectFeaturesLabels(dbConf)
    logger.info("feature vectors: %s", len(X))
    X_normed = StandardScaler().fit_transform(X)

    pcaVarianceExplained = []
    ldaVarianceExplained = []
    componentsStart = 2
    componentsStop = 55  # using around 63 features total
    components = list(range(componentsStart, componentsStop))
    for c in components:
        # TODO what are the rules of thumb for number of components for clustering
        # TODO similar what is a good amount of variance capture to go ahead and use?
        pca = PCA(n_components=c)
        s = time.time()
        pca.fit_transform(X_normed)
        logger.info("pca in %s", time.time() - s)
        pcaVe = sum(pca.explained_variance_ratio_)
        pcaVarianceExplained.append(pcaVe)

        lda = LinearDiscriminantAnalysis(n_components=c)
        s = time.time()
        lda.fit_transform(X_normed, y)
        logger.info("lda in %s", time.time() - s)
        ldaVe = sum(lda.explained_variance_ratio_)
        ldaVarianceExplained.append(ldaVe)
        logger.info("components: %s PCA: %s LDA: %s", c, pcaVe, ldaVe)


    # TODO look into the guts of LDA to see what we can take away
    plt.semilogy(components, pcaVarianceExplained, label="PCA (unlabeled)")
    plt.semilogy(components, ldaVarianceExplained, label="LDA (labeled)")
    plt.xlabel("components")
    plt.ylabel("variance explained")
    plt.title("Variance explained vs. components")
    plt.grid(True)
    plt.legend(loc="lower right")
    plt.show()


if __name__ == "__main__":
    main()
