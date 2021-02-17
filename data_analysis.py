#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

OUTPUT_PATH = "Thesis/SimulationOutput/"
PROPAGATION_DAT = "CR3BPnormalisedCoRotatingFrame.dat"
MASS_PARAMETER = 0.000953684

def parse_state(file_name):
    state_dict = dict()

    # Parse .dat file given by PROPAGATION_DAT
    with open(OUTPUT_PATH + PROPAGATION_DAT) as file_data:

        for line in iter(file_data.readline, b''):
            # Stop if empty string
            if not line:
                break
            line_split = line[:-2].split(", ")
            time = line_split[1]
            state_array = np.array(line_split[1:])
            # state dictionary, key: time, value: state
            state_dict.update({time : state_array})

    return state_dict

def state_dict_to_mat(state_dict):
    n_samples = len(state_dict.keys())
    state_mat = np.zeros((n_samples, 6))
    for count, state_iter in enumerate(state_dict.values()):
        state_mat[count,:] = state_iter.transpose()

    return state_mat


state_dict = parse_state(PROPAGATION_DAT)
state_mat = state_dict_to_mat(state_dict)

plt.plot(state_mat[:,0], state_mat[:,1])
plt.plot(1 - MASS_PARAMETER, 0, 'o', markersize=7)
plt.plot(MASS_PARAMETER, 0, 'o', markersize=14)
plt.show()