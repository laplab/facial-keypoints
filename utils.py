import numpy as np
import csv


def read_coords(filename, offset=0):
    res = {}
    with open(filename) as file:
        reader = csv.reader(file)
        next(reader)
        for row in reader:
            coords = np.array([float(x) for x in row[1 + offset:]], dtype='float64')
            res[row[offset]] = coords
    return res


def center_circle(X):
    return (X - X.mean(axis=0)) / X.std(axis=0)
