import numpy as np
import pandas as pd
from scipy.spatial import KDTree

# The radius of points to return.
radius = 50
# Which Minkowski p-norm to use. Should be in the range [1, inf].
p_norm = 2 #float('inf')
# Approximate search. Branches of the tree are not explored if their nearest points are further than r / (1 + eps), and branches are added in bulk if their furthest points are nearer than r * (1 + eps).
eps = 1
# The number of nearest neighbors to return.
k = 5

def find_nearest(restaurants_kd_tree, r, hotel):
    idx = restaurants_kd_tree.query_ball_point(hotel, r=r, p=p_norm)
    return len(idx)

def find_score_for_positions(positions, r, restaurants_tree):

    print 'looping through positions..'
    for position in positions:
        print position
        print find_nearest(restaurants_tree, r, position)

def find_knn(tree, position, k_points):
    return tree.query(position, k=k_points)

def find_knn_score_for_positions(positions, k_points, restaurants_tree):
    print 'looping through positions..'
    positions_data = list()
    for position in positions:
        distances, indices = find_knn(restaurants_tree, position, k_points=k_points)
        if(len(distances) > 0):
            positions_data.append(distances[-1])
    return positions_data

print 'loading data..'

restaurants = pd.read_csv('C:\Users\cfilip09\Downloads\\vlahou\\restaurants-ver1.txt', sep="|", header=1, names = ['id','name','rating','latitude','longitude','cousine']).as_matrix()
hotels = pd.read_csv('C:\Users\cfilip09\Downloads\\vlahou\hotels-ver1.txt', sep="|", header=1, names = ['id','name','rating','rating1','latitude','longitude','cousine']).as_matrix()

print 'indexing..'
tree = KDTree(restaurants[:, [3, 4]])

print 'task 1..'
input_hotels = hotels[:, [4, 5]][:3]
find_score_for_positions(input_hotels, radius, tree)
print 'task 1 done..'


print 'task 2..'
print find_knn_score_for_positions(input_hotels, k, tree)
print 'task 2 done..'

# print sorted(hotels_score, key=lambda a_entry: a_entry[1], reverse=True)[:50]
#for hotel in hotels:
#    idx = tree.query([hotel[3], hotel[4]], k, p=)
