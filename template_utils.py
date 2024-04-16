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