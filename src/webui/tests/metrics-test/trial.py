import random
import time

import nni
import numpy as np


def fetch_metric(a):
    if a == 'normal':
        return random.uniform(-10, 10)
    elif a == 'inf':
        return float('inf')
    elif a == 'neginf':
        return float('-inf')
    elif a == 'nan':
        return float('nan')
    elif a == 'string':
        return str(random.uniform(-10, 10))
    elif a == 'dict-normal':
        return {'default': random.uniform(-10, 10), 'other': random.uniform(-10, 10)}
    elif a == 'dict-nodefault':
        return {'other': random.uniform(-10, 10)}
    elif a == 'dict-empty':
        return {}
    elif a == 'dict-defaultdict':
        return {'default': {'tensor': 0, 'data': random.uniform(-10, 10)}}
    elif a == 'numpy':
        return np.random.uniform()
    else:
        raise AssertionError


params = nni.get_next_parameter()
for i in range(params['intermediate_count']):
    time.sleep(1)
    nni.report_intermediate_result(fetch_metric(params['intermediate' + str(i + 1)]))
for i in range(params['final_count']):
    time.sleep(1)
    nni.report_final_result(fetch_metric(params['final' + str(i + 1)]))
