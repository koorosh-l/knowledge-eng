import math
import csv
import os
import numpy as np
from pprint import pprint

data_set_root = '/home/koorosh/proj/knowledge-eng/data/'
output_csv    = './output.csv'
window_size   = 60

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
        segments = {}
        for seg in os.listdir(os.path.join(so_far, "p1")):
            path = os.path.join(so_far, "p1", seg)
            dist = func(path)
            segment = int(seg.split(".")[0][1:])
            segments[segment] = dist
        act = int(dir[1:])
        res[act] = segments
    return res

data = for_all_files_do(data_set_root, file_processor)

def querry(act, seg):
    return data[act][seg]

def mysplit(dict, p=.8):
    train_size  = int(60 * p)
    test_size   = 60-train_size
    train_label = []
    train_data  = []
    test_label  = []
    test_data   = []
    #train
    for i in range(1,20):
        for j in range(1,train_size):
            train_label.append(i)
            train_data.append(dict[i][j])
    #test
    for i in range(1,20):
        for j in range(train_size,61):
            test_label.append(i)
            test_data.append(dict[i][j])
    return train_data, train_label, test_data, test_label

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

X_train, y_train, X_test, y_test = mysplit(data, p=.9)

svm_classifier = SVC(kernel='rbf')
svm_classifier.fit(X_train, y_train)
svm_predictions = svm_classifier.predict(X_test)
svm_accuracy = accuracy_score(y_test, svm_predictions)


print("SVM Accuracy:", svm_accuracy)


knn_classifier = KNeighborsClassifier(n_neighbors=3)
knn_classifier.fit(X_train, y_train)
knn_predictions = knn_classifier.predict(X_test)
knn_accuracy = accuracy_score(y_test, knn_predictions)


print("KNN Accuracy:", knn_accuracy)
