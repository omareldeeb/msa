import numpy as np
import itertools
import operator
import sys
import heapq
import math

'''
Dot function as defined in the distance function (see Readme)
'''
def dot(seqs, x, b):
    k = len(seqs)
    return tuple([mask(seqs[i][x[i] - 1], b[i]) for i in range(0, k)])


def mask(c, b):
    if b == 0:
        return '-'
    else:
        return c

'''
Generic cost function for k sequences;
distance 2 for gaps,
distance 3 for mismatches
'''
def cost(seq):
    k = len(seq)
    sum = 0
    for i in range(0, k):
        for j in range(i, k):
            if seq[i] == seq[j]:
                sum += 0
            elif seq[i] == '-' or seq[j] == '-':
                sum += 2
            else:
                sum += 3  # set to 5 to penalize mismatches more harshly
    return sum


def sequence_distance(seqs):
    # memoize matrix initialization
    k = len(seqs)
    init_x = np.zeros(k, dtype=int)
    dim = tuple(map(lambda seq: len(seq) + 1, seqs))  # matrix dimensions
    mem = np.full(dim, np.inf, dtype=int)
    mem[init_x] = 0

    bit_vecs = list(itertools.product([0, 1], repeat=k))[1:]  # all k bit vectors from (0...1) to (1...1)
    vecs = [np.zeros(k, dtype=int)]  # list with vectors to pop for calculation
    heapq.heapify(vecs)
    while len(vecs) is not 0:
        curr_vec = heapq.heappop(vecs)
        for bit_vec in bit_vecs:
            index_vec = list(map(operator.add, curr_vec, bit_vec))  # curr_vec + bit_vec gives us our next index vector
            if (np.array(index_vec) >= np.array(dim)).any():  # skip vector if index goes out of bounds
                continue
            if index_vec not in vecs:
                heapq.heappush(vecs, index_vec)
                val = mem.item(tuple(curr_vec)) + cost(dot(seqs, index_vec, bit_vec))
                mem[tuple(index_vec)] = val
            else:
                val = min(mem.item(tuple(index_vec)), mem.item(tuple(curr_vec)) + cost(dot(seqs, index_vec, bit_vec)))
                mem[tuple(index_vec)] = val

    # output last matrix entry for final distance and return memoization matrix
    print('Calculated distance: ' + str(mem.item((tuple(map(lambda x: x - 1, dim))))))
    return mem


def find_optimal_path(mem):
    optimal_path = []
    dim = mem.shape
    curr_vec = tuple(x - 1 for x in dim)
    curr_dist = mem.item(tuple(x - 1 for x in dim))
    bit_vecs = list(itertools.product([0, 1], repeat=k))[1:]

    while curr_dist != 0:
        for bit_vec in bit_vecs:
            index_vec = list(map(operator.sub, curr_vec, bit_vec))
            if any(x < 0 for x in index_vec):
                continue
            dist = mem.item(tuple(index_vec))
            if int(curr_dist - dist) == int(cost(dot(seqs, curr_vec, bit_vec))):
                # print(bit_vec)
                optimal_path.append(bit_vec)
                curr_dist = dist
                curr_vec = list(map(operator.sub, curr_vec, bit_vec))

    return optimal_path


def print_sequence(path):
    print_strings = []
    for step in reversed(path):
        for i in range(0, k):
            if len(print_strings) <= i:
                print_strings.append("")
            if step[i] == 0:
                print_strings[i] += '-'
            else:
                print_strings[i] += list(reversed(seqs[i])).pop()
                seqs[i] = seqs[i][1:]

    for string in print_strings:
        print(string)


# Credits go to Andrew Schwartz:
# https://stackoverflow.com/questions/25494668/best-way-to-plot-a-3d-matrix-in-python/25512111
def cube_marginals(cube, normalize=False):
    c_fcn = np.mean if normalize else np.sum
    xy = c_fcn(cube, axis=0)
    xz = c_fcn(cube, axis=1)
    yz = c_fcn(cube, axis=2)
    return (xy, xz, yz)


# Based off of Andrew Schwartz's Answer:
# https://stackoverflow.com/questions/25494668/best-way-to-plot-a-3d-matrix-in-python/25512111
def plotcube(zlabel, ylabel, xlabel, cube, x=None, y=None, z=None, normalize=False, plot_front=False):
    """Use contourf to plot cube marginals"""
    (Z, Y, X) = cube.shape
    (xy, xz, yz) = cube_marginals(cube, normalize=normalize)
    if x is None: x = np.arange(X)
    if y is None: y = np.arange(Y)
    if z is None: z = np.arange(Z)

    fig = plot.figure()
    ax = fig.gca(projection='3d')

    # draw edge marginal surfaces
    offsets = (Z - 1, 0, X - 1) if plot_front else (0, Y - 1, 0)
    cset = ax.contourf(x[None, :].repeat(Y, axis=0), y[:, None].repeat(X, axis=1), xy, zdir='z', offset=offsets[0],
                       cmap=plot.cm.coolwarm, alpha=0.75)
    cset = ax.contourf(x[None, :].repeat(Z, axis=0), xz, z[:, None].repeat(X, axis=1), zdir='y', offset=offsets[1],
                       cmap=plot.cm.coolwarm, alpha=0.75)
    cset = ax.contourf(yz, y[None, :].repeat(Z, axis=0), z[:, None].repeat(Y, axis=1), zdir='x', offset=offsets[2],
                       cmap=plot.cm.coolwarm, alpha=0.75)

    # draw wire cube to aid visualization
    ax.plot([0, X - 1, X - 1, 0, 0], [0, 0, Y - 1, Y - 1, 0], [0, 0, 0, 0, 0], 'k-')
    ax.plot([0, X - 1, X - 1, 0, 0], [0, 0, Y - 1, Y - 1, 0], [Z - 1, Z - 1, Z - 1, Z - 1, Z - 1], 'k-')
    ax.plot([0, 0], [0, 0], [0, Z - 1], 'k-')
    ax.plot([X - 1, X - 1], [0, 0], [0, Z - 1], 'k-')
    ax.plot([X - 1, X - 1], [Y - 1, Y - 1], [0, Z - 1], 'k-')
    ax.plot([0, 0], [Y - 1, Y - 1], [0, Z - 1], 'k-')

    ax.set_xlabel(xlabel)
    xlen = len(xlabel)
    plot.xlim(0, xlen)
    ax.set_xticks([i for i in range(0, xlen+1)])

    ax.set_ylabel(ylabel)
    ylen = len(ylabel)
    plot.ylim(0, ylen)
    ax.set_yticks([i for i in range(0, ylen+1)])

    ax.set_zlabel(zlabel)
    zlen = len(zlabel)
    ax.set_zlim(0, zlen)
    ax.set_zticks([i for i in range(0, zlen+1)])

    plot.show()


if __name__ == "__main__":
    k = len(sys.argv) - 1
    if k <= 1:
        print("Supply at least 2 sequences")
        exit(1)

    seqs = sys.argv[1:]
    mem = sequence_distance(seqs)
    print_sequence(find_optimal_path(mem))
    # plot a cube in the case of a 3D matrix
    if k == 3:
        import matplotlib.pyplot as plot
        import mpl_toolkits.mplot3d.axes3d as axes3d
        plotcube(sys.argv[1], sys.argv[2], sys.argv[3], mem)
