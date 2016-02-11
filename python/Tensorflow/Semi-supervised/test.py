import numpy as np


prediction_indices = [[a, 0], [1860, 1], [835, 0]]
prediction_indices = sorted(prediction_indices, key=lambda row: row[0])
print prediction_indices