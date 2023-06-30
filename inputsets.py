import numpy as np


def get_input_set(input_count):
    input_sets = []
    inputs = 2 ** input_count
    for i in range(inputs):
        variable_count = input_count
        in_set = list(bin(i)[2:].zfill(variable_count))
        in_set = np.array(in_set)
        s = []
        for j in in_set:
            s.append(int(j))
        input_sets.append(s)
    return input_sets
