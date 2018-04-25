#!/usr/bin/env python3
from sklearn import cluster, datasets


def m():
    iris = datasets.load_iris()
    xIris = iris.data
    yIris = iris.target

    model = cluster.KMeans(n_clusters=3)
    model.fit(xIris)
    clustering = model.labels_[::10]
    print(clustering)
    print(yIris[::10])


if __name__=="__main__":
    m()
