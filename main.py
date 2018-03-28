import numpy as np
import pandas as pd
from scipy.spatial import KDTree
import itertools

# The radius of points to return.
radius = 50
# Which Minkowski p-norm to use. Should be in the range [1, inf].
p_norm = 2 #float('inf')
# Approximate search. Branches of the tree are not explored if their nearest points are further than r / (1 + eps), and branches are added in bulk if their furthest points are nearer than r * (1 + eps).
eps = 1
# The number of nearest neighbors to return.
k = 5

def find_nearest(restaurants_kd_tree, r, hotel):
    idx = restaurants_kd_tree.query_ball_point([hotel[1], hotel[2]], r=r, p=p_norm)
    return [hotel[0], len(idx)]

def find_score_for_positions(positions, r, restaurants_tree):
    positions_data = list()
    for position in positions:
        positions_data.append(find_nearest(restaurants_tree, r, position))
    return positions_data

def find_knn(tree, hotel, k_points):
    ind, distance = tree.query([hotel[1], hotel[2]], k=k_points)
    return [hotel[0], distance[-1]]

def find_knn_score_for_positions(hotels, k_points, restaurants_tree):
    positions_data = list()
    
    for hotel in hotels:
        knn = find_knn(restaurants_tree, hotel, k_points=k_points)
        positions_data.append(knn)

    return positions_data

# loading data..
hotels = pd.read_csv('C:\Users\cfilip09\Downloads\\vlahou\hotels-ver1.txt', sep="|", header=1, names = ['id','name','rating','rating1','latitude','longitude','cousine']).as_matrix()
restaurants = pd.read_csv('C:\Users\cfilip09\Downloads\\vlahou\\restaurants-ver1.txt', sep="|", header=1, names = ['id','name','rating','latitude','longitude','cousine']).as_matrix()

# indexing
tree = KDTree(restaurants[:, [3, 4]])

np.random.shuffle(hotels)
input_hotels = hotels[:, [0, 4, 5]][:3]

# task 1..
score_positions = find_score_for_positions(input_hotels, radius, tree)
print sorted(score_positions, key=lambda x: x[1])

# task 2..
knn_score_positions = find_knn_score_for_positions(input_hotels, k, tree)
print sorted(knn_score_positions, key=lambda x: x[1])

print '---------------------'
print input_hotels[:, [1, 2]]
dd = itertools.combinations(tuple([tuple(row) for row in input_hotels[:, [1, 2]]]), 2)
print set(dd)

