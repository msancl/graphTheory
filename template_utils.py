# If needed, write here your additional fuctions/classes with their signature and use them in the exercices:
# a specific place is available to copy them at the end of the Inginious task.

# First, import the libraries needed for your helper functions
import numpy as np
import pandas as pd

# Then write the classes and/or functions you wishes to use in the exercises
def example_helper_function(arg1, arg2):
    return 0

def bridges_count(graph):
    V = len(graph)
    visited = [False] * (V + 1) 
    disc = [float("Inf")] * (V + 1)
    low = [float("Inf")] * (V + 1)
    parent = [-1] * (V + 1)
    Time = 0
    bridges = []

    def bridgeUtil(u):
        nonlocal Time
        visited[u] = True
        disc[u] = low[u] = Time
        Time += 1

        for v in graph[u]:
            if not visited[v]:
                parent[v] = u
                bridgeUtil(v)
                low[u] = min(low[u], low[v])
                if low[v] > disc[u]:
                    bridges.append((u, v))
            elif v != parent[u]:
                low[u] = min(low[u], disc[v])

    for i in graph:
        if not visited[i]:
            bridgeUtil(i)

    return len(bridges)

def find_local_bridges(graph):
    local_bridges = 0 

    for u in graph.keys():
        for v in graph[u]:
            if u < v:

                common_neighbors = set(graph[u]) & set(graph[v])
                common_neighbors.discard(u)  
                common_neighbors.discard(v)  

                if not common_neighbors:
                    local_bridges+=1

    return local_bridges

def create_graph(dataframe):
    graph = {}

    for a, b in dataframe.itertuples(index=False):
        if a not in graph:
            graph[a] = []
        if b not in graph:
            graph[b] = []
        graph[a].append(b)
        graph[b].append(a)

    return graph
    