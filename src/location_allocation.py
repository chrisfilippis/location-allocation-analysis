import numpy as np
import pandas as pd
import itertools
from scipy.spatial import KDTree
from scipy.spatial import cKDTree
import os
import time
import matplotlib.pyplot as plt

# Which Minkowski p-norm to use. Should be in the range [1, inf].
p_norm = 2
leafsize = 10

def find_nearest_core(kd_tree, r, point, pnorm=2):
    idx = kd_tree.query_ball_point([point[1], point[2]], r=r, p=pnorm)
    return idx

def find_nearest(kd_tree, r, point, pnorm=2):
    idx = find_nearest_core(kd_tree, r, point, p_norm)
    return [point[0], len(idx)]

def find_score_for_positions(positions, r, restaurants_tree):
    positions_data = list()
    for position in positions:
        positions_data.append(find_nearest(restaurants_tree, r, position))
    return positions_data

def find_knn(tree, hotel, k_points):
    ind, distance = tree.query([hotel[1], hotel[2]], k=k_points)
    return [hotel[0], max(ind)]

def find_knn_score_for_positions(hotels, k_points, restaurants_tree):
    positions_data = list()
    
    for hotel in hotels:
        knn = find_knn(restaurants_tree, hotel, k_points=k_points)
        positions_data.append(knn)

    return positions_data

def get_points_combinations(points):
    unique_combinations = set(itertools.combinations(tuple([tuple(row) for row in points]), 2))
    return [np.array(list(point_comb)) for point_comb in unique_combinations]

def find_neibhor_points(combinations, kd_tree, radius, max_distance):
    result = list()

    for index in range(len(combinations)):
        combination = combinations[index]
        lat_lon_combination = combination[:,[1,2]]
        combination_points_distance = np.linalg.norm(lat_lon_combination[0]-lat_lon_combination[1])

        if(combination_points_distance > max_distance):
            continue

        first_point_nearest = find_nearest_core(kd_tree, radius, combination[0])
        second_point_nearest = find_nearest_core(kd_tree, radius, combination[1])
        score = len(set().union(first_point_nearest, second_point_nearest))
        result.append([combination[0][0], combination[1][0], score])

    if(len(result) == 0):
        return []

    return sorted(result, key=lambda x: x[2])

def load_data():
    # loading data..
    relative_path = os.path.abspath(os.path.dirname(__file__))

    hotels = pd.read_csv(os.path.join(relative_path, '../data/hotels-ver1.txt'), sep="|", header=1, names = ['id','name','rating','rating1','latitude','longitude','cousine']).as_matrix()
    restaurants = pd.read_csv(os.path.join(relative_path, '../data/restaurants-ver1.txt'), sep="|", header=1, names = ['id','name','rating','latitude','longitude','cousine']).as_matrix()
    return hotels, restaurants

def create_tree(restaurants):
    return cKDTree(restaurants[:, [3, 4]], leafsize=leafsize)

def choose_random_hotels(hotels_data, n):
    # np.random.shuffle(hotels_data)
    if(n == -1):
        return hotels_data[:, [0, 4, 5]]
    else:
        return hotels_data[:, [0, 4, 5]][:n]

def task_1(tree, hotels, radius, hotels_number, save_results=False):
    # select random points
    input_hotels = choose_random_hotels(hotels, hotels_number)

    start = time.time()
    # task 1..
    score_positions = find_score_for_positions(input_hotels, radius, tree)
    sorted_scores = sorted(score_positions, key=lambda x: x[1], reverse=True)
    task_time = time.time() - start

    if(save_results == True):
        relative_path = os.path.abspath(os.path.dirname(__file__))
        filename = os.path.join(relative_path, "../data/results/task1/task_" + str(hotels_number) + "_" + str(radius) + ".csv")
        np.savetxt(fname=filename, X=sorted_scores, delimiter=",", fmt='%i', header="id,score", comments='')

    return task_time

def task_2(tree, hotels, k, hotels_number, save_results=False):
    # select random points
    input_hotels = choose_random_hotels(hotels, hotels_number)
    
    start = time.time()
    # task 2..
    knn_score_positions = find_knn_score_for_positions(input_hotels, k, tree)
    sorted_knn_score_positions = sorted(knn_score_positions, key=lambda x: x[1])
    task_time = time.time() - start

    if(save_results == True):
        relative_path = os.path.abspath(os.path.dirname(__file__))
        filename = os.path.join(relative_path, "../data/results/task2/task_" + str(hotels_number) + "_" + str(k) + ".csv")
        np.savetxt(fname=filename, X=sorted_knn_score_positions, delimiter=",", fmt='%i', header="id,score", comments='')

    return task_time

def task_3(tree, hotels, radius, hotels_number, save_results=False):
    # select random points
    input_hotels = choose_random_hotels(hotels, hotels_number)

    start = time.time()

    input_combinations = np.array(get_points_combinations(input_hotels))
    max_distance = (2 * radius)
    result = find_neibhor_points(input_combinations, tree, radius, max_distance)
    task_time = time.time() - start

    if(len(result) < 1 or len(result[0]) == 0):
        return []

    if(save_results == True):
        save_result = np.array(result)
        relative_path = os.path.abspath(os.path.dirname(__file__))
        filename = os.path.join(relative_path, "../data/results/task3/task_" + str(hotels_number) + "_" + str(radius) + ".csv")
        np.savetxt(fname=filename, X=save_result[:,[0, 1]], delimiter=",", fmt='%i', header="point1,point2", comments='')
    
    best_combination = [result[0][0], result[0][1]] 

    return [task_time, best_combination]

def tests_for_m_task1(tree, hotels):
    m_tests = [5, 10, 15]
    results = list()
    for test in m_tests:
        results.append([test, task_1(tree, hotels, 1, test, True)])

    print results
    data = np.array(results)
    plt.plot(data[:,0], data[:,1])
    plt.xlabel('number of hotels')
    plt.ylabel('time (sec)')
    plt.show()
    return np.array(results)

def tests_for_radius_task1(tree, hotels):
    radius_tests = [1, .1, .01, .001]
    results = list()
    for test in radius_tests:
        results.append([test, task_1(tree, hotels, test, -1, True)])

    print results
    data = np.array(results)
    plt.plot(data[:,0], data[:,1])
    plt.xlabel('radius')
    plt.ylabel('time (sec)')
    plt.show()
    return np.array(results)

def tests_for_m_task2(tree, hotels):
    m_tests = [500, 1000, 1500, 2000, 5000]
    results = list()
    for test in m_tests:
        results.append([test, task_2(tree, hotels, 5, test, True)])

    print results
    data = np.array(results)
    plt.plot(data[:,0], data[:,1])
    plt.xlabel('number of hotels')
    plt.ylabel('time (sec)')
    plt.show()
    return np.array(results)

def tests_for_k_task2(tree, hotels):
    radius_tests = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22]
    results = list()
    for test in radius_tests:
        results.append([test, task_2(tree, hotels, test, -1, True)])

    print results
    data = np.array(results)
    plt.plot(data[:,0], data[:,1])
    plt.xlabel('k')
    plt.ylabel('time (sec)')
    plt.show()
    return np.array(results)

def tests_for_task3(tree, hotels):
    radius_tests = [1, .1, .01, .001]
    results = list()
    for test in radius_tests:
        data = task_3(tree, hotels, test, 100, True)
        if(len(data) == 0):
            return []
        results.append([test, data[0], data[1]])

    print results

hotels, restaurants = load_data()
kd_tree = create_tree(restaurants)

tests_for_m_task1(kd_tree, hotels)
#tests_for_radius_task1(kd_tree, hotels)
#tests_for_m_task2(kd_tree, hotels)
#tests_for_k_task2(kd_tree, hotels)
#tests_for_task3(kd_tree, hotels)