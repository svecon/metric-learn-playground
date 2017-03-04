import numpy as np
import math
# import pandas as pd
# import glob
# import os.path
# import sys

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

def commonStyles(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ax.tick_params(axis='x', direction='out')
    ax.tick_params(axis='y', length=0)

    ax.grid(axis='y', color="0.9", linestyle='-', linewidth=1)
    ax.set_axisbelow(True)

def startGraphing(title=None, cols=1, N=1, size=None, sharey=False):
    fig = plb.figure()
    if title:
        st = fig.suptitle(title, fontsize=12)
    
    rows = math.ceil(N/cols)

    if size is None:
        fig.set_size_inches(min(5.70866, 4*rows), min(9.72441, 3*cols), forward=False)
    else:
        fig.set_size_inches(*size, forward=False)
    
    axes = []
    for i in range(N):
        sharey_val = axes[0] if sharey and len(axes)>0 else None
        axes.append(fig.add_subplot(rows,cols,i+1,sharey=sharey_val))

    return fig, axes

def endGraphing(fig, legend=None, filename=None, move_title=0.825, legend_ncol=3, adjust_legend=0.175, legend_position='bottom'):
    if move_title:
        fig.subplots_adjust(top=move_title)

    # Shrink current axis by 20%
    # box = ax.get_position()
    # ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    # fig.subplots_adjust(right=0.80)

    plt.tight_layout(pad=0.0, w_pad=1.0, h_pad=3.0)

    if legend is not None:
        # Right of the plot: loc='center left', bbox_to_anchor=(1, 0.5));
        if legend_position=='bottom':
            legend = fig.legend(fig.get_axes()[0].lines, legend, ncol=legend_ncol, bbox_to_anchor=(.0, 0.0, 1., 0.), loc='lower left', mode="expand", borderaxespad=0.)
            fig.subplots_adjust(bottom=adjust_legend)
        elif legend_position=='right':
            legend = fig.legend(fig.get_axes()[0].lines, legend, ncol=legend_ncol, bbox_to_anchor=(1,1), loc='upper left')
            fig.subplots_adjust(right=adjust_legend)
        
        frame = legend.get_frame()        
        frame.set_facecolor('1.0')
        frame.set_edgecolor('1.0')

    if filename is not None:
        fig.savefig(filename+'.pdf')    

def plotBox(ax, data, labels, means=None, title=None, xlabel=None, ylabel='successrate', doubleColors=False, skipped=None, rotateLabels=90):
    if skipped is None:
        skipped = [0]*len(labels)
    
    if title is not None:
    	ax.set_title(title, y=1.025)

    if ylabel: ax.set_ylabel(ylabel, size=10)
    if xlabel: ax.set_xlabel(xlabel, size=10)
    
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

    commonStyles(ax)
    ax.set_xticklabels(labels, rotation=rotateLabels)
            
    # if ylabel[:11]=='successrate':
        # plt.ylim(ymax=1.0)

    # fig.set_size_inches(min(10, 0.66*len(labels)), 4, forward=True)
    

def plotLines(ax, data, x_ticks, labels=None, title=None, xlabel=None, ylabel='successrate', doubleColors=False, rotateLabels=0):
    ax.set_xticks(np.arange(len(x_ticks)))
    ax.set_xticklabels(x_ticks, rotation=rotateLabels)
    
    if title: ax.set_title(title, y=1.025)
    
    if ylabel: ax.set_ylabel(ylabel, size=10)
    if xlabel: ax.set_xlabel(xlabel, size=10)

    colors = generateColors(alpha=255, doubleColors=doubleColors)

    for i,X in enumerate(data):
        ax.plot(np.arange(len(X)), X, linewidth=2, color=colors[i])

    if labels is not None:
	    legend = plb.legend(labels, loc='center left', bbox_to_anchor=(1, 0.5));
	    frame = legend.get_frame()
	    frame.set_facecolor('1.0')
	    frame.set_edgecolor('1.0')
    
    # Shrink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    commonStyles(ax)
    
    # if ylabel[:11]=='successrate':
        # plt.ylim(ymax=1.0)


def plotFitness(ax, fitnesses, best_results, worst_results, mean_results, baseline, min_gen=0, max_gen=None, title=None, xlabel=None, ylabel='successrate', **kwargs):
    bmap = brewer2mpl.get_map('Set2', 'qualitative', 7)
    colors = bmap.mpl_colors
     
    if max_gen is not None:
        fitnesses = fitnesses[min_gen:max_gen]
        best_results = best_results[min_gen:max_gen]
        worst_results = worst_results[min_gen:max_gen]
        mean_results = mean_results[min_gen:max_gen]

    x = np.arange(1+min_gen, fitnesses.shape[0]+1+min_gen)

    med, perc_25, perc_75 = perc(fitnesses)
    maxs = np.max(fitnesses, axis=1)
    mins = np.min(fitnesses, axis=1)

    if title:
        ax.set_title(title, y=1.05)
    
    if ylabel: ax.set_ylabel(ylabel, size=10)
    if xlabel: ax.set_xlabel(xlabel, size=10)

    ax.plot(x, gaussian_filter1d(maxs, sigma=2.0, axis=0), linewidth=1, color=colors[1])
    ax.fill_between(x, perc_25, perc_75, alpha=0.25, linewidth=0, color=colors[0]) 
    ax.plot(x, gaussian_filter1d(med, sigma=2.0, axis=0), linewidth=1, color=colors[0])

    ax.plot(x, gaussian_filter1d(best_results, sigma=2.0, axis=0), linewidth=2, color=colors[2])
    ax.plot(x, gaussian_filter1d(worst_results, sigma=2.0, axis=0), linewidth=2, color=colors[3])
    # ax.plot(x, gaussian_filter1d(mean_results, sigma=2.0, axis=0), linewidth=2, color=colors[4])

    ax.plot(x, gaussian_filter1d(mins, sigma=2.0, axis=0), linewidth=1, color=colors[1])
    ax.plot(x, [baseline]*len(x), linewidth=1, color='black')

    commonStyles(ax)
    
    if ylabel[:11]=='successrate':
        # plt.ylim(ymax=1.0)
        ax.set_ylim([0.0, 1.0])


def plotScatter(ax, title, X_train, y_train, X_test, y_test, wrong, score, xlabel=None, ylabel=None):
    colors = generateColors(alpha=255, doubleColors=False, skipped=None)

    X_train = X_train.T
    X_test = X_test.T

    ax.set_title('{} ({:.2f}%)'.format(title, score*100), y=1.05)
    
    def fsizes(a):
        return [x*50+75 for x in a]
    
    def fcolors(a):
        return [colors[x] for x in a]
    
    def fedge(a):
        return [(1,0,0,1) if x else (0,0,0,0) for x in a]
    
    ax.scatter(X_train[0], X_train[1], s=fsizes(np.zeros(len(y_train))), color=fcolors(y_train), alpha=0.15)
    ax.scatter(X_test[0], X_test[1], s=fsizes(wrong), color=fcolors(y_test), alpha=0.55, edgecolors=fedge(wrong))

    ax.set_xticklabels([])
    ax.set_yticklabels([])

    commonStyles(ax)

    ax.spines['bottom'].set_visible(False)
    ax.grid(axis='x', color="0.9", linestyle='-', linewidth=0)
    ax.grid(axis='y', color="0.9", linestyle='-', linewidth=0)
    ax.tick_params(axis='x', length=0)
