import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

def path_cost(x, y, accumulated_cost, distances):
    path = [[len(x)-1, len(y)-1]]
    cost = 0
    i = len(y)-1
    j = len(x)-1
    while i>0 and j>0:
        if i==0:
            j = j - 1
        elif j==0:
            i = i - 1
        else:
            if accumulated_cost[i-1, j] == min(accumulated_cost[i-1, j-1], accumulated_cost[i-1, j], accumulated_cost[i, j-1]):
                i = i - 1
            elif accumulated_cost[i, j-1] == min(accumulated_cost[i-1, j-1], accumulated_cost[i-1, j], accumulated_cost[i, j-1]):
                j = j-1
            else:
                i = i - 1
                j= j- 1
        path.append([j, i])
    path.append([0,0])
    for [y, x] in path:
        cost = cost +distances[x, y]
    return path, cost

def calculate_cost(x, y):
    distances = np.zeros((len(y), len(x)))

    for i in range(len(y)):
        for j in range(len(x)):
            distances[i,j] = (x[j]-y[i])**2

    accumulated_cost = np.zeros((len(y), len(x)))

    for i in range(1, len(x)):
        accumulated_cost[0,i] = distances[0,i] + accumulated_cost[0, i-1] 

    for i in range(1, len(y)):
        accumulated_cost[i,0] = distances[i, 0] + accumulated_cost[i-1, 0] 

    for i in range(1, len(y)):
        for j in range(1, len(x)):
            accumulated_cost[i, j] = min(accumulated_cost[i-1, j-1], accumulated_cost[i-1, j], accumulated_cost[i, j-1]) + distances[i, j]

    path, cost = path_cost(x, y, accumulated_cost, distances)

    return cost

def find_centroid(labels, indices):
    assert len(labels) == len(indices)
    indices_hash = {}
    for i in xrange(len(labels)):
        if labels[i] == -1 or len(indices[i]) > 2:
            continue
        if indices_hash.has_key(labels[i]):
            indices_hash[labels[i]].append(indices[i])
        else:
            indices_hash[labels[i]] = [indices[i]]
    for k, v in indices_hash.iteritems():
        indices_hash[k] = [sum(l)/len(l) for l in zip(*v)]
    return indices_hash.values()

def find_start_end_indices(left_models, right_models, df):
    window_sizes = [45, 60]
    COST_THRESHOLD = 300

    numbers = []
    left_indices = []
    right_indices = []

    for index, row in df.iterrows():
        print index
        curr_theta = row['theta']
        numbers.append(curr_theta)

        if len(numbers) < window_sizes[-1]:
            continue

        if len(numbers) > window_sizes[-1]:
            numbers = numbers[1:]

        for l in left_models:
            min_cost = float("inf")
            w_size = 0
            for w in window_sizes:
                left_curr_cost = calculate_cost(l, numbers[-w:])
                if left_curr_cost < min_cost:
                    min_cost = left_curr_cost
                    w_size = w
            if min_cost < COST_THRESHOLD and w_size > 0:
                left_indices.append([index - w_size, index])
                break
        for r in right_models:
            min_cost = float("inf")
            w_size = 0
            for w in window_sizes:
                right_curr_cost = calculate_cost(r, numbers[-w:])
                if right_curr_cost < min_cost:
                    min_cost = right_curr_cost
                    w_size = w
            if min_cost < COST_THRESHOLD and w_size > 0:
                right_indices.append([index - w_size, index])
                break

    if len(left_indices) > 0:
        left_db = DBSCAN(eps=20).fit(left_indices)
        left_indices = find_centroid(left_db.labels_, left_indices)

    if len(right_indices) > 0:
        right_db = DBSCAN(eps=20).fit(right_indices)
        right_indices = find_centroid(right_db.labels_, right_indices)

    return { "left lane change start": [x[0] for x in left_indices],
             "left lane change end": [x[1] for x in left_indices], 
             "right lane change start": [x[0] for x in right_indices],
             "right lane change end": [x[1] for x in right_indices], }



