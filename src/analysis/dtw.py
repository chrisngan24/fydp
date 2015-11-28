import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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

def find_start_end_indices(model, df):

    numbers = []
    min_cost = float("inf")
    best_window_index = 0
    best_window_size = 0
    prev_theta = 0
    threshold = 0.005
    min_window = 20
    flag = 0

    window_size = 60

    for index, row in df.iterrows():
        curr_theta = row['theta']
        numbers.append(curr_theta)
        # delta = abs(curr_theta) - abs(prev_theta)
        if len(numbers) < window_size:
            continue

        if len(numbers) > window_size:
            numbers = numbers[1:]

        curr_cost = calculate_cost(model, numbers)
        if curr_cost < min_cost:
            best_window_index = index - window_size
            # best_window_index = index - len(numbers)
            # best_window_size = len(numbers)
            min_cost = curr_cost

        prev_theta = curr_theta

    return { "start": best_window_index, "end": best_window_index + window_size }

