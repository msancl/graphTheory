# If needed, write here your additional fuctions/classes with their signature and use them in the exercices:
# a specific place is available to copy them at the end of the Inginious task.

# First, import the libraries needed for your helper functions
import numpy as np
import pandas as pd

# Then write the classes and/or functions you wishes to use in the exercises
def example_helper_function(arg1, arg2):
    return 0

def pagerank(graph, graph_reversed, N, d=0.85, maxerr=0.000001):

    pr_score = {}
    
    for i in range(1, N+1):
        pr_score[i] = (1-d)/N
    
    loop = 0
    pr_biggest = 0
    pr_biggest_num = 0
    
    while True:
        pr_biggest = 0

        loop +=1
        err = 0
        pr_score_new = {}
        for i in range(1, N+1):
            pr_score_new[i] = (1-d)/N
            for j in graph_reversed[i]:
                pr_score_new[i] += d * pr_score[j] / len(graph[j])
            err += abs(pr_score_new[i] - pr_score[i])
            if(pr_biggest<pr_score_new[i]):
                pr_biggest = pr_score_new[i]
                pr_biggest_num = i
        pr_score = pr_score_new
        if err < maxerr:
            break
        
    return pr_biggest, pr_biggest_num

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

def create_unidirectional_graph(dataframe):
    graph = {}
    graph_reversed = {}

    N = 0
    for a, b in dataframe.itertuples(index=False):
        if a not in graph:
            graph[a] = []
            graph_reversed[a] = []
        if b not in graph:
            graph[b] = []
            graph_reversed[b] = []

        graph[a].append(b)
        graph_reversed[b].append(a)
    return graph, graph_reversed




def BFS(graph, start):
    visited = set()
    queue = [(start, 0)]
    shortest_path_count = {}

    while queue:
        node, distance = queue.pop(0)
        if node not in visited:
            visited.add(node)
            shortest_path_count[distance] = shortest_path_count.get(distance, 0) + 1
            for neighbor in graph[node]:
                if neighbor not in visited:
                    queue.append((neighbor, distance + 1))

    return shortest_path_count