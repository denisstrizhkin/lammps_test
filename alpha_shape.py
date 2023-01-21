#!python3

import os
os.environ['OPENBLAS_NUM_THREADS'] = '12'

import numpy as np
import alphashape
from descartes import PolygonPatch
import matplotlib.pyplot as plt

def main():
    data_path='./results/fall.dump'
    data = np.loadtxt(fname=data_path, skiprows=9)
    
    alpha_shape = alphashape.alphashape(data, 0.1)

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot_trisurf(*zip(*alpha_shape.vertices), triangles=alpha_shape.faces)
    plt.show()

if __name__ == '__main__':
    main()