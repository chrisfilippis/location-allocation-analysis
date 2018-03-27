import numpy as np
import pandas as pd
from scipy.spatial import KDTree

radius = 50
k = 5

restaurants = pd.read_csv('C:\Users\cfilip09\Downloads\\vlahou\\restaurants-ver1.txt', sep="|", header = None, names = ['id','name','rating','latitude','longitude','cousine']).as_matrix()
hotels = pd.read_csv('C:\Users\cfilip09\Downloads\\vlahou\hotels-ver1.txt', sep="|", header = None, names = ['id','name','rating','rating1','latitude','longitude','cousine']).as_matrix()

tree = KDTree(restaurants[:, [3, 4]],)

for hotel in hotels:
    idx = tree.query_ball_point([hotel[3], hotel[4]], r=radius, p=float('inf'))



#for hotel in hotels:
#    idx = tree.query([hotel[3], hotel[4]], k, p=)
