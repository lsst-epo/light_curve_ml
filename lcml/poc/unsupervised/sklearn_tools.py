

# Hierarchical clustering

# - sklearn.cluster.AgglomerativeClustering (bottom-up)

"""
- Ward minimizes the sum of squared differences within all clusters.
It is a variance-minimizing approach and in this sense is similar to the
k-means objective function but tackled with an agglomerative hierarchical
approach.
- Average linkage minimizes the average of the distances between all
observations of pairs of clusters.
- Maximum or complete linkage minimizes the maximum distance between
observations of pairs of clusters.

Feature Agglomeration
sklearn.cluster.FeatureAgglomeration

__Varying the metric__
Average and complete linkage can be used with a variety of distances
(or affinities), in particular Euclidean distance (l2), Manhattan distance
(or Cityblock, or l1), cosine distance, or any precomputed affinity matrix.

- l1 distance is often good for sparse features, or sparse noise: ie many
of the features are zero, as in text mining using occurrences of rare words.
- cosine distance is interesting because it is invariant to global scalings of
 the signal.


"""

"""
The FeatureAgglomeration uses agglomerative clustering to group together 
features that look very similar, thus decreasing the number of features. 
"""