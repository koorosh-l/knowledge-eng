import math
from pprint import pprint

def eu_dist(v1, v2):
    res = 0
    for i in  range(len(v1)):
        res += (v1[i] - v2[i]) ** 2
    return math.sqrt(res)
def vector_profile(input, sub_arr):
    res = []
    ws = len(sub_arr)
    for i in range(len(input) - ws + 1):
        v1 = input[i:i+ws]
        # print(v1)
        # print(sub_arr)
        res.append(eu_dist(sub_arr, v1))
    return res
def matrix_profile(vec, n):
    res = [];
    for i in range(0,len(vec) - n + 1):
        sub_arr = vec[i:i+n]
        res.append(vector_profile(vec, sub_arr))
        # print(sub_arr)
    return res
