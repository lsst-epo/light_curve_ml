from sklearn import cluster, datasets


def m():
    iris = datasets.load_iris()
    k_means = cluster.KMeans(n_clusters=3)
    xIris = iris.data
    yIris = iris.target

    k_means.fit(xIris)
    clustering = k_means.labels_[::10]
    print(clustering)
    print(yIris[::10])

if __name__=="__main__":
    m()
