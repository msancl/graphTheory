import pandas as pd
import numpy as np
import sys 
import matplotlib.pyplot as plt
from template_utils import *
import networkx as nx
import time



sys.setrecursionlimit(6000)

# Undirected graph
# Task 1: Average degree, number of bridges, number of local bridges
def Q1(dataframe):
    df = pd.DataFrame({'Connections': [0] * 4942})
    for a,b in dataframe.itertuples(index=False):
        df.at[a, 'Connections'] += 1
        df.at[b, 'Connections'] += 1
    
    mean = df['Connections'][1:].mean()


    plt.figure(figsize=(10, 5))
    max_degree = df['Connections'][1:].max()
    bins = range(0, max_degree + 2)
    plt.hist(df['Connections'][1:], bins=bins, color='blue', edgecolor='black')
    plt.title('Histogram of Degrees', fontsize=14)
    plt.xlabel('Degree', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.savefig('histogram_of_degrees.pdf', format='pdf')
    plt.show()

    graph = create_graph(dataframe)

    bridges = bridges_count(graph)
    local_bridges = find_local_bridges(graph)

    return [mean, bridges, local_bridges] # [average degree, nb bridges, nb local bridges]

# Undirected graph
# Task 2: Average similarity score between neighbors
def Q2(dataframe):

    graph = create_graph(dataframe)

    similarity_scores = []
    num_neighbors = []

    for node, neighbors in graph.items():
        for i in neighbors:
            if(i<node):
                continue
            common_neighbors = set(neighbors) & set(graph[i])
            total_neighbors = set(neighbors) | set(graph[i])
            sim_score = len(common_neighbors) / (len(total_neighbors)-2)
            similarity_scores.append(sim_score)


     
    
    average_similarity = np.sum(similarity_scores) / len(similarity_scores)
    
    similarity_scores.sort()
    
    print("similarity_scores: ",similarity_scores)

    edge_percentage = np.linspace(0, 100, len(similarity_scores))

    plt.plot(similarity_scores, edge_percentage)
    plt.xlabel('Similarity Score')
    plt.ylabel('Percentage of Edges')
    plt.title('Percentage of Edges vs. the Similarity Score')
    plt.savefig('percentage_of_edges_vs_the_similarity_score.pdf', format='pdf')
    plt.show()
    
    
    return average_similarity 




    


# Directed graph
# Task 3: PageRank
def Q3(dataframe):
    # Your code here
    
    graph, graph_reversed = create_unidirectional_graph(dataframe)
    
    N = len(graph)
    
    
    print(graph[428], N)    
    print(graph_reversed[428], N)
    pr_biggest, pr_biggest_num = pagerank(graph, graph_reversed, N)
    
    return [pr_biggest_num, pr_biggest] # the id of the node with the highest pagerank score, the associated pagerank value.
    # Note that we consider that we reached convergence when the sum of the updates on all nodes after one iteration of PageRank is smaller than 10^(-6)






# Undirected graph
# Task 4: Small-world phenomenon
def Q4(dataframe):
    # Your code here
    
    graph = create_graph(dataframe)
    
    shortest_path_count = {}
    
    for node in graph:
        paths = BFS(graph, node)
        for distance, count in paths.items():
            shortest_path_count[distance] = shortest_path_count.get(distance, 0) + count
    
    # shortest_path_count = {0: 4941, 1: 13188, 2: 32070, 3: 60992, 4: 104216, 5: 161518, 6: 231116, 7: 317050, 8: 417178, 9: 527538, 10: 643300, 11: 760572, 12: 876378, 13: 993332, 14: 1106938, 15: 1212646, 16: 1303336, 17: 1364872, 18: 1387570, 19: 1388020, 20: 1371436, 21: 1333408, 22: 1280458, 23: 1222186, 24: 1151852, 25: 1063390, 26: 944232, 27: 800454, 28: 648234, 29: 499750, 30: 366986, 31: 260126, 32: 179052, 33: 121462, 34: 84140, 35: 59208, 36: 42164, 37: 30202, 38: 20678, 39: 12908, 40: 7356, 41: 4008, 42: 1918, 43: 738, 44: 260, 45: 88, 46: 16}
    
    
    shortest_path_list = [0] * (max(shortest_path_count.keys())+1)
    average_distance = 0
    amount_of_paths = 0
    for distance, count in shortest_path_count.items():
        shortest_path_list[distance] = count
        average_distance+=distance*count
        amount_of_paths+=count
    average_distance /= amount_of_paths
        
    distances = range(1, len(shortest_path_list) + 1)
    num_paths = shortest_path_list

    plt.bar(distances, num_paths)
    plt.xlabel('Distance')
    plt.ylabel('Number of Paths')
    plt.title('Number of Shortest Paths by Distance')
    plt.yticks(range(0, int(max(num_paths))+1, int(max(num_paths)/10)))
    plt.savefig('number_of_shortest_paths_by_distance.pdf', format='pdf')
    plt.show()
    
    print("avgdistances = ", np.average(average_distance))
    
    return shortest_path_list # at index 0 the number of shortest paths of lenght 0, at index 1 the number of shortest paths of length 1, ...
    # Note that we will ignore the value at index 0 as it can be set to 0 or the number of nodes in the graph

# Undirected graph
# Task 5: Betweenness centrality
def Q5(dataframe):
    # Your code here
    return [0, 0.0] # the id of the node with the highest betweenness centrality, the associated betweenness centrality value.

# you can write additionnal functions that can be used in Q1-Q5 functions in the file "template_utils.py", a specific place is available to copy them at the end of the Inginious task.


df = pd.read_csv('powergrid.csv')
# print("Q1", Q1(df))
# print("Q2", Q2(df))
# print("Q3", Q3(df))

start_time = time.time()

print("Q4", Q4(df))
# print("Q5", Q5(df))

end_time = time.time()
execution_time = end_time - start_time
print("Execution time:", execution_time, "seconds")
