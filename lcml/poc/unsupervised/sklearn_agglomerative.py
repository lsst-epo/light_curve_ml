from sklearn import datasets
from sklearn.cluster import AgglomerativeClustering


def main():
    iris = datasets.load_iris()
    xIris = iris.data
    yIris = iris.target

    # these should suffice for feets features
    nClusters = 3
    linkage = "ward"
    affinity = "euclidean"
    kwargs = {"n_clusters": nClusters, "linkage": linkage, "affinity": affinity}
    model = AgglomerativeClustering(n_clusters=nClusters, **kwargs)
    model.fit(xIris)

    clustering = model.labels_[::10]
    print(clustering)
    print(yIris[::10])


if __name__ == "__main__":
    main()
