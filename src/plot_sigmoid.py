import matplotlib.pyplot as plt
import numpy as np

import math

def sigmoid(x):
    a = []
    for item in x:
        a.append(1/(1+math.exp(-item)))
    return a

def build_cartesian_plane(max_quadrant_range, add_x_dashes=False):
    """ The quadrant range controls the range of the quadrants"""
    l = []
    zeros = []
    plt.grid(True, color='b', zorder=0,)
    ax = plt.axes()
    head_width = float(0.05) * max_quadrant_range
    head_length = float(0.1) * max_quadrant_range
    ax.arrow(0, 0, max_quadrant_range, 0, head_width=head_width, head_length=head_length, fc='k', ec='k',zorder=100)
    ax.arrow(0, 0, -max_quadrant_range, 0, head_width=head_width, head_length=head_length, fc='k', ec='k', zorder=100)
    ax.arrow(0, 0, 0, max_quadrant_range, head_width=head_width, head_length=head_length, fc='k', ec='k', zorder=100)
    ax.arrow(0, 0, 0, -max_quadrant_range, head_width=head_width, head_length=head_length, fc='k', ec='k', zorder=100)
    if add_x_dashes:
        counter_dash_width = max_quadrant_range * 0.005
        dividers = [0, .1,.2,.3,.4, .5, .6, .7, .8, .9, 1]
        for i in dividers:
            plt.plot([-counter_dash_width, counter_dash_width], [i*max_quadrant_range, i*max_quadrant_range], color='k')
            plt.plot([i * max_quadrant_range, i*max_quadrant_range], [-counter_dash_width, counter_dash_width], color='k')
            plt.plot([-counter_dash_width, counter_dash_width], [-i * max_quadrant_range, -i * max_quadrant_range], color='k')
            plt.plot([-i * max_quadrant_range, -i * max_quadrant_range], [-counter_dash_width, counter_dash_width], color='k')
            l.append(i * max_quadrant_range)
            l.append(-i * max_quadrant_range)
            zeros.append(0)
            zeros.append(0)


build_cartesian_plane(10)

def plot_sigmoid():
    x = np.arange(-10., 10., 0.2)
    sig = sigmoid(x)

    fig = plt.figure()
    plt.ylim(ymin=-0.5, ymax=1.5)

    build_cartesian_plane(10)
    plt.plot(x,sig, 'b')
    fig.savefig('./Sigmoid.jpg')
    fig.clf()

def plot_svm_example():
    # evenly sampled time at 200ms intervals
    t = np.random.normal(0., 1., 20)
    y = np.random.normal(0., 1., 20)

    t_1 = np.random.normal(4., 1., 20)
    y_2 = np.random.normal(4., 1., 20)


    fig = plt.figure()
    plt.ylim(ymin=-0.5, ymax=1.5)
    build_cartesian_plane(10)

    # red dashes, blue squares and green triangles
    plt.plot(t_1, y, 'r^', t, y_2, 'g^')
    fig.savefig('./SVM_ilustration.jpg')
    fig.clf()

# plot_svm_example()
plot_sigmoid()