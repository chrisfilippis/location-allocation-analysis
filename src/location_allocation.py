import numpy as np
import pandas as pd
import itertools
from scipy.spatial import KDTree
from scipy.spatial import cKDTree
import os
import time

# Which Minkowski p-norm to use. Should be in the range [1, inf].
p_norm = 2 #float('inf')

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

def choose_random_hotels(hotels_data, n):
    np.random.shuffle(hotels_data)
    if(n == -1):
        return hotels_data[:, [0, 4, 5]]
    else:
        return hotels_data[:, [0, 4, 5]][:n]

def task_1(radius, k, hotels_number):
    # loading data..
    hotels, restaurants = load_data()

    # select random points
    input_hotels = choose_random_hotels(hotels, hotels_number)

    start = time.time()

    # indexing
    tree = cKDTree(restaurants[:, [3, 4]], leafsize=10)

    start = time.time()
    # task 1..
    score_positions = find_score_for_positions(input_hotels, radius, tree)
    sorted(score_positions, key=lambda x: x[1])
    task_time = time.time() - start

    return task_time

def task_2(k, hotels_number):
    # loading data..
    hotels, restaurants = load_data()

    # select random points
    input_hotels = choose_random_hotels(hotels, hotels_number)

    start = time.time()

    # indexing
    tree = cKDTree(restaurants[:, [3, 4]], leafsize=10)

    start = time.time()
    # task 2..
    knn_score_positions = find_knn_score_for_positions(input_hotels, k, tree)
    sorted(knn_score_positions, key=lambda x: x[1])
    task_time = time.time() - start

    return task_time

def task_3(radius, k, hotels_number):
    # loading data..
    hotels, restaurants = load_data()

    # select random points
    input_hotels = choose_random_hotels(hotels, hotels_number)

    start = time.time()

    # indexing
    tree = cKDTree(restaurants[:, [3, 4]], leafsize=10)

    start = time.time()

    input_combinations = np.array(get_points_combinations(input_hotels))
    max_distance = (2 * radius)
    result = find_neibhor_points(input_combinations, tree, radius, max_distance)
    task_time = time.time() - start

    return task_time, result

def run_all_task(radius, k, hotels_number):
    # loading data..
    hotels, restaurants = load_data()

    # select random points
    input_hotels = choose_random_hotels(hotels, hotels_number)

    start = time.time()

    # indexing
    tree = cKDTree(restaurants[:, [3, 4]], leafsize=10)

    start = time.time()
    # task 1..
    score_positions = find_score_for_positions(input_hotels, radius, tree)
    sorted(score_positions, key=lambda x: x[1])
    task1_time = time.time() - start

    start = time.time()
    # task 2..
    knn_score_positions = find_knn_score_for_positions(input_hotels, k, tree)
    sorted(knn_score_positions, key=lambda x: x[1])
    task2_time = time.time() - start

    start = time.time()
    # task 3..
    input_combinations = np.array(get_points_combinations(input_hotels))
    max_distance = (2 * radius)
    find_neibhor_points(input_combinations, tree, radius, max_distance)
    task3_time = time.time() - start

    return task1_time, task2_time, task3_time

def tests_for_m_task1():
    m_tests = [500, 1000, 1500, 2000, -1]
    results = list()
    for test in m_tests:
        results.append([test, task_1(1, 10, test)])

    print results

def tests_for_radius_task1():
    radius_tests = [1, .1, .01, .001]
    results = list()
    for test in radius_tests:
        results.append([test, task_1(test, 10, -1)])

    print results

def tests_for_m_task2():
    m_tests = [500, 1000, 1500, 2000, -1]
    results = list()
    for test in m_tests:
        results.append([test, task_2(5, test)])

    print results

def tests_for_k_task2():
    radius_tests = [2, 4, 6, 8, 10]
    results = list()
    for test in radius_tests:
        results.append([test, task_2(test, -1)])

    print results

tests_for_m_task1()
tests_for_radius_task1()
tests_for_m_task2()
tests_for_k_task2()