#!python3

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

LATTICE = 5.43
SQUARE = LATTICE / 2

def get_linspace(left, right):
    return np.linspace(left, right, round((right - left) / SQUARE) + 1)

def main():
    data_path='./results/fall.dump'

    data = np.loadtxt(fname=data_path, skiprows=9)
    dimensions = np.loadtxt(fname=data_path, skiprows=5, max_rows=3)

    X = get_linspace(dimensions[0, 0], dimensions[0, 1])
    Y = get_linspace(dimensions[1, 0], dimensions[1, 1])
    Z = np.zeros((len(X), len(Y)))
    
    for i in range(len(X) - 1):
        for j in range(len(Y) - 1):
            Z_vals = data[np.where((data[:,0] >= X[i]) & (data[:,0] < X[i + 1]) & (data[:,1] >= Y[j]) & (data[:,1] < Y[j + 1]))][:,2]

            if len(Z_vals) == 0:
                Z[i, j] = np.nan
            else:
                Z[i, j] = Z_vals.max()

            #print(Z[i,j])

    print(f'NaN: {np.count_nonzero(np.isnan(Z))}')
    Z[np.where(np.isnan(Z))] = np.nanmean(Z)

    n_X = Z.shape[0]
    X = np.linspace(0, n_X - 1, n_X, dtype=int)

    n_Y = Z.shape[1]
    Y = np.linspace(0, n_Y - 1, n_Y, dtype=int)

    def f_Z(i, j):
        return Z[i,j]
    
    Xs, Ys = np.meshgrid(X, Y)
    Z = f_Z(Xs, Ys)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(Xs, Ys, Z, cmap=cm.jet)
    plt.savefig('teste.pdf')
    plt.show()


if __name__ == '__main__':
    main()