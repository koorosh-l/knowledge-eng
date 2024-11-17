import math
import csv
import os
import numpy as np
from pprint import pprint

data_set_root = 'data/'
window_size   = 10

#matrix_profile
def eu_dist(v1, v2):
    res = 0
    for i in  range(len(v1)):
        res += (v1[i] - v2[i]) ** 2
    return math.sqrt(res)

def vector_min(input_vector, window):
    min = float("+inf")
    window_size = len(window)
    for i in range(len(input_vector)-window_size+1):
        sub = input_vector[i:i+window_size]
        res = eu_dist(sub, window)
        if res < min and res != 0:
            min = res
    return min

def distance_matrix(input_vector, window_size):
    res = []
    for i in range(0,len(input_vector)-window_size+1):
        window = input_vector[i:i+window_size]
        res.append(vector_min(input_vector, window))
    return res

#data processing
def get_first_column(file):
    with open(file, "r") as file:
        reader = csv.reader(file)
        return [float(row[0]) for row in reader]

def file_processor(file):
    return distance_matrix(get_first_column(file), window_size)

def make_label(dir,seg):
    return dir+seg.split(".")[0]

def for_all_files_do(root, func):
    res = {}
    for dir in os.listdir(root):
        so_far = os.path.join(root, dir)
        for seg in os.listdir(os.path.join(so_far, "p1")):
            label = make_label(dir, seg)
            file = os.path.join(so_far, "p1", seg)
            res[label] = func(file)
    return res

dict = for_all_files_do(data_set_root, file_processor)

def querry(dir, seg):
    return dict[make_label(dir, seg)]
