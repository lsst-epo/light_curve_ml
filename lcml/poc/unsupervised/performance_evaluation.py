"""
A. Given the knowledge of the ground truth class assignments

- Adjusted Rand index
from sklearn import metrics
metrics.adjusted_rand_score(labels_true, labels_pred)

- Mutual Information based scores
from sklearn import metrics
metrics.adjusted_mutual_info_score(labels_true, labels_pred)
metrics.normalized_mutual_info_score(labels_true, labels_pred)

- V-Measure
sklearn.metrics.homogeneity_completeness_v_measure
homogeneity: each cluster contains only members of a single class.
completeness: all members of a given class are assigned to the same cluster.
V = f(homogeneity, completeness)
V == normalized_mutual_info !!

NB. prefer adjusted rand score and adjusted mutual information for samples
less than 1,000 or clusters > 10

- sklearn.metrics.fowlkes_mallows_score

B. No labled data

- Silhouette Coefficient
kmeans_model = KMeans(n_clusters=3, random_state=1).fit(X
labels = kmeans_model.labels_
metrics.silhouette_score(X, labels, metric='euclidean')

Calinski-Harabaz Index
metrics.calinski_harabaz_score(X, labels)
"""