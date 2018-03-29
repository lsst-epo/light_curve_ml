import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler

def testIris():
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


import matplotlib.pyplot as plt

from lcml.pipeline.database.sqlite_db import selectFeaturesLabels
from lcml.utils.context_util import jsonConfig
def main():
    conf = jsonConfig("pipeline.json")
    dbConf = conf["database"]
    dbConf["dbPath"] = "data/macho/macho_processed.db"

    X, y = selectFeaturesLabels(dbConf)
    X_normed = StandardScaler().fit_transform(X)
    components = []
    pcaVarianceExplained = []
    ldaVarianceExplained = []
    for c in range(5, 45):
        # TODO what are the rules of thumb for number of components for clustering
        # TODO similar what is a good amount of variance capture to go ahead and use?
        pca = PCA(n_components=c)
        pca.fit(X_normed)
        X_r = pca.transform(X_normed)
        components.append(c)
        pcaVarianceExplained.append(sum(pca.explained_variance_ratio_))

        lda = LinearDiscriminantAnalysis(n_components=c)
        X_r2 = lda.fit(X_normed, y).transform(X_normed)
        ldaVarianceExplained.append(sum(lda.explained_variance_ratio_))

    # TODO look into the guts of LDA to see what we can take away
    plt.plot(components, pcaVarianceExplained)
    plt.plot(components, ldaVarianceExplained)
    plt.xlabel("components")
    plt.ylabel("variance explained")
    plt.title("Variance explained vs. components")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()
