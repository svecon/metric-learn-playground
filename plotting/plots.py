import numpy as np
# import pandas as pd
# import glob
# import os.path
# import sys
# import math

# from svecon.HierarchicalGridSearchCV import HierarchicalGridSearchCV
# from svecon.EmptyTransformer import EmptyTransformer

# from sklearn.cross_validation import StratifiedKFold
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import Imputer, StandardScaler, LabelEncoder

# %matplotlib inline
import pylab as plb
import matplotlib
import matplotlib.pyplot as plt
import brewer2mpl

from scipy.ndimage.filters import gaussian_filter1d

def stars(p):
    if p < 0.0001:
        return "****"
    elif (p < 0.001):
        return "***"
    elif (p < 0.01):
        return "**"
    elif (p < 0.05):
        return "*"
    else:
        return "-"

def generateColors(alpha=255, doubleColors=False, skipped=None):
#     colors = brewer2mpl.get_map('Set3', 'qualitative', min(12, len(labels))).mpl_colors
    colors = [(31, 119, 180, alpha), (255, 127, 14, alpha), (70, 171, 70, alpha), (214, 39, 40, alpha), (148, 103, 189, alpha),
          (140, 86, 75, alpha), (227, 119, 194, alpha), (127, 127, 127, alpha), (188, 189, 34, alpha), (23, 190, 207, alpha)]
    colors = [[y/255.0 for y in x] for x in colors]
    colors = colors+colors
    if doubleColors:
        colors = [val for val in colors for _ in (0, 1)]
    
    if skipped is not None:
        for i in sorted(skipped, reverse=True):
            del colors[i]
            
    return colors

def perc(data):
    median = np.zeros(data.shape[0])
    perc_25 = np.zeros(data.shape[0])
    perc_75 = np.zeros(data.shape[0])
    for i in range(0, len(median)):
        median[i] = np.median(data[i, :])
        perc_25[i] = np.percentile(data[i, :], 25)
        perc_75[i] = np.percentile(data[i, :], 75)
    return median, perc_25, perc_75

params = {
#     predefined https://github.com/jbmouret/matplotlib_for_papers
    'axes.labelsize': 8,
    'font.size': 8,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'text.usetex': False,
#     'figure.figsize': [4.5, 4.5],
# my own
    'figure.titlesize': 10,
    'axes.titlesize': 10,
    'axes.labelsize': 8,
    'figure.figsize': [7, 4],
   }
matplotlib.rcParams.update(params)

def commonStyles(fig, ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ax.tick_params(axis='x', direction='out')
    ax.tick_params(axis='y', length=0)

    ax.grid(axis='y', color="0.9", linestyle='-', linewidth=1)
    ax.set_axisbelow(True)

    fig.subplots_adjust(left=0.2)
#     fig.subplots_adjust(bottom=0.15)
    plt.tight_layout()

def plotBox(title, data, labels, means=None, xlabel=None, ylabel='successrate (%)', doubleColors=False, skipped=None, rotateLabels=90):
    if skipped is None:
        skipped = [0]*len(labels)
    
    fig = plb.figure()
    ax = fig.add_subplot(111)
    ax.set_title(title, y=1.05)

    if ylabel: plb.ylabel(ylabel, size=10)
    if xlabel: plb.xlabel(xlabel, size=10)
    
    bp = ax.boxplot(data, notch=0, sym='b+', vert=1, whis=1.5, positions=None, widths=0.6, showmeans=False)
    
    if means is not None:
        x = np.arange(1, len(labels)+1)
        [mp] = ax.plot(x, means)
        mp.set_color('red')

    colors = generateColors(alpha=128, doubleColors=doubleColors, skipped=skipped)
        
    for i in range(len(bp['boxes'])):
        box = bp['boxes'][i]
        box.set_linewidth(0)
        boxX = []
        boxY = []
        for j in range(5):
            boxX.append(box.get_xdata()[j])
            boxY.append(box.get_ydata()[j])
            boxPolygon = plb.Polygon(list(zip(boxX,boxY)), facecolor = colors[i % len(colors)], linewidth=0)
            ax.add_patch(boxPolygon)

    for i in range(0, len(bp['boxes'])):
        bp['boxes'][i].set_color(colors[i])
        # we have two whiskers!
        bp['whiskers'][i*2].set_color(colors[i])
        bp['whiskers'][i*2 + 1].set_color(colors[i])
        bp['whiskers'][i*2].set_linewidth(2)
        bp['whiskers'][i*2 + 1].set_linewidth(2)
        # top and bottom fliers
        bp['fliers'][i].set(markerfacecolor=colors[i], marker='o', alpha=0.75, markersize=6, markeredgecolor='none')
#         bp['fliers'][i * 2 + 1].set(markerfacecolor=colors[i], marker='o', alpha=0.75, markersize=6, markeredgecolor='none')
        bp['medians'][i].set_color('black')
        bp['medians'][i].set_linewidth(3)
        # and 4 caps to remove
        for c in bp['caps']:
            c.set_linewidth(2)

    commonStyles(fig, ax)
    ax.set_xticklabels(labels, rotation=rotateLabels)
            
    if ylabel[:11]=='successrate':
        plt.ylim(ymax=1.0)

    fig.set_size_inches(min(10, 0.66*len(labels)), 4, forward=True)
    
#     plb.savefig('boxplot4.pdf')

def plotLines(title, data, labels, x_ticks, xlabel=None, ylabel='successrate (%)', doubleColors=False):
    fig = plb.figure()
    ax = fig.add_subplot(111)
    ax.set_xticks(np.arange(len(x_ticks)))
    ax.set_xticklabels(x_ticks, rotation=0)
    ax.set_title(title, y=1.05)
    
    if ylabel: plb.ylabel(ylabel, size=10)
    if xlabel: plb.xlabel(xlabel, size=10)

    colors = generateColors(alpha=255, doubleColors=doubleColors)

    for i,X in enumerate(data):
        plt.plot(np.arange(len(X)), X, linewidth=2, color=colors[i])

    legend = plb.legend(labels, loc='center left', bbox_to_anchor=(1, 0.5));
    frame = legend.get_frame()
    frame.set_facecolor('1.0')
    frame.set_edgecolor('1.0')
    
    # Shrink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    commonStyles(fig, ax)
    
    if ylabel[:11]=='successrate':
        plt.ylim(ymax=1.0)

    plt.tight_layout()
#     plb.savefig('boxplot4.pdf')

def plotFitness(title, data, bestresults, worstresults, xlabel=None, ylabel='successrate (%)'):
    bmap = brewer2mpl.get_map('Set2', 'qualitative', 7)
    colors = bmap.mpl_colors
     
    x = np.arange(1, data.shape[0]+1)

    med, perc_25, perc_75 = perc(data)
    maxs = np.max(data, axis=1)
    mins = np.min(data, axis=1)

    fig = plb.figure() # no frame
    ax = fig.add_subplot(111)
    ax.set_title(title, y=1.05)
    
    if ylabel: plb.ylabel(ylabel, size=10)
    if xlabel: plb.xlabel(xlabel, size=10)

    ax.plot(x, gaussian_filter1d(maxs, sigma=2.0, axis=0), linewidth=1, color=colors[1])
    ax.fill_between(x, perc_25, perc_75, alpha=0.25, linewidth=0, color=colors[0]) 
    ax.plot(x, gaussian_filter1d(med, sigma=2.0, axis=0), linewidth=1, color=colors[0])

    ax.plot(x, gaussian_filter1d(bestresults, sigma=2.0, axis=0), linewidth=2, color=colors[2])
    ax.plot(x, gaussian_filter1d(worstresults, sigma=2.0, axis=0), linewidth=2, color=colors[3])

    ax.plot(x, gaussian_filter1d(mins, sigma=2.0, axis=0), linewidth=1, color=colors[1])

    legend = ax.legend(["Maximal fitness", "Median fitness", "Successrate using best", "Successrate using worst","Minimal fitness"], loc='center left', bbox_to_anchor=(1, 0.5));
    frame = legend.get_frame()
    frame.set_facecolor('1.0')
    frame.set_edgecolor('1.0')

    # Shrink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    commonStyles(fig, ax)
    
    if ylabel[:11]=='successrate':
        plt.ylim(ymax=1.0)

    plt.tight_layout()
    # fig.savefig('variance_matplotlib.png')
