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

def find_score_for_positions(positions, r, restaurants_data):
    tree = KDTree(restaurants_data[:, [3, 4]])

    print 'looping through positions..'

    for position in positions:
        print position
        print find_nearest(tree, r, position)

restaurants = pd.read_csv('C:\Users\cfilip09\Downloads\\vlahou\\restaurants-ver1.txt', sep="|", header=1, names = ['id','name','rating','latitude','longitude','cousine']).as_matrix()
hotels = pd.read_csv('C:\Users\cfilip09\Downloads\\vlahou\hotels-ver1.txt', sep="|", header=1, names = ['id','name','rating','rating1','latitude','longitude','cousine']).as_matrix()

print 'indexing..'
input_hotels = hotels[:, [4, 5]][:3]
find_score_for_positions(input_hotels, radius, restaurants)
print 'ready..'

# print sorted(hotels_score, key=lambda a_entry: a_entry[1], reverse=True)[:50]
#for hotel in hotels:
#    idx = tree.query([hotel[3], hotel[4]], k, p=)
