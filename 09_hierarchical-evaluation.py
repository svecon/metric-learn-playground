
# coding: utf-8

# Hierarchical evaluation
# ========

# In[ ]:

#use full width of screen in Jupyter notebooks
# from IPython.core.display import display, HTML
# display(HTML("<style>.container { width:100% !important; }</style>"))

import numpy as np
import pandas as pd
import glob
import os.path
import sys
import math

from hierarchical_grid_search_cv.HierarchicalGridSearchCV import HierarchicalGridSearchCV
from hierarchical_grid_search_cv import EmptyTransformer

from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Imputer, StandardScaler, LabelEncoder

from metric_learn import LMNN, NCA, LFDA, Covariance, MetricEvolution #, CMAES, FullMatrixTransformer, NeuralNetworkTransformer
from metric_learn import ITML_Supervised, SDML_Supervised, LSML_Supervised, RCA_Supervised

datasetsDirectory = 'datasets'
resultsDirectory = 'results/classification'
dumpsDirectory = 'results/dumps'

if not os.path.exists(resultsDirectory):
    os.makedirs(resultsDirectory)

if not os.path.exists(dumpsDirectory):
    os.makedirs(dumpsDirectory)

default_n_jobs = 8
default_random_state = 789
default_n_folds = 10
default_shuffle = True

import logging
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
logger.handlers = []
fh = logging.FileHandler("{}/error.log".format(resultsDirectory))
fh.setLevel(logging.DEBUG)
logger.addHandler(fh)
fh = logging.StreamHandler(sys.stdout)
fh.setLevel(logging.DEBUG)
logger.addHandler(fh)

import pickle
def save_object(obj, filename):
    with open("{}/dump_{}.bin".format(dumpsDirectory,filename), 'wb') as output:
        logger.info("saving file" + filename)
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)


# In[ ]:

cv_per_dataset = {}
def evaluateClassifier(X, y, pipeline, parameters, name=None, datasetName=None):
    
    if datasetName in cv_per_dataset:
        cv = cv_per_dataset[datasetName]
    else:
        cv_per_dataset[datasetName] = cv =             StratifiedKFold(n_splits=default_n_folds, shuffle=default_shuffle, random_state=default_random_state)
    
    grid_search = HierarchicalGridSearchCV(pipeline, parameters, n_jobs=default_n_jobs, verbose=4, cv=cv)
    grid_search.fit(X, y)
    
    save_object(grid_search, '{}_{}'.format(name,datasetName))
    
    stats = [{
        **x,
        **x['scores'],
        **x['train_scores'],
        **x['params'],
        **x['times'],
     } for x in grid_search.grid_scores_ ]

    for i in stats:
        i.pop('scores')
        i.pop('train_scores')
        i.pop('params')
        i.pop('times')
            
    df = pd.DataFrame(stats)
    df['technique'] = pd.Series([name]*df.shape[0], index=df.index)
    df['dataset'] = pd.Series([datasetName]*df.shape[0], index=df.index)
    
    return df

def evaluatePipeline(X,y,datasetName,pipeline):
    resFilename = '{}/{}_{}_result.csv'.format(resultsDirectory,datasetName,pipeline.__name__[8:])
    
    if os.path.isfile(resFilename):
        logging.info("\t`{}` using `{}` already finished, skipping".format(datasetName,pipeline.__name__[8:]))
        return None
    
    logging.info("\t`{}` using `{}` started".format(datasetName,pipeline.__name__[8:]))
    res = pipeline(X,y,datasetName)
    res.to_csv(resFilename)
    logging.info("\t`{}` using `{}` finished".format(datasetName,pipeline.__name__[8:]))

def noConstraints(Y):
    c = len(set(Y))
    return (10*c*c, 20*c*c, 40*c*c, 80*c*c)


# In[ ]:

defaultKnnParams = {
    'knn__n_neighbors': (1, 2, 4, 8, 16, 32, 64, 128),
}
defaultIters = (50, 250, 500, 1000)
defaultImputer = Imputer(missing_values='NaN', strategy='mean', axis=0, verbose=0, copy=False)
defaultStandardizer = StandardScaler(copy=False, with_mean=True, with_std=True)

paramsCmaes = {
    'cmaes__transformer': ('full', 'diagonal'),
    'cmaes__s__n_gen': (50, 100, 250),
    'cmaes__f__n_neighbors': (1, 4, 8),#, 16),
    # 'cmaes__c__weights': ('uniform', 'distance'),
}

paramsCmaesFme = {
    'cmaesfme__transformer': ('full', 'diagonal'),
    'cmaesfme__s__n_gen': (50, 100, 250),
}

paramsJdeKnn = {
    'jdeknn__transformer': ('diagonal',), #('full', 'diagonal'),
    'jdeknn__s__n_gen': (25, 100),
    'jdeknn__f__n_neighbors': (1, 4, 8)#, 16),
    # 'jdeknn__f__weights': ('uniform', 'distance'),
}

paramsJde = {
    'jde__transformer': ('diagonal',), #('full', 'diagonal'),
    'jde__s__n_gen': (25, 100),
}

paramsJdePur = {
    'jdepur__transformer': ('diagonal',), #('full', 'diagonal'),
    'jdepur__s__n_gen': (25, 100),
    'jdepur__f__sig': (0.5, 1, 10),#, 50),
}

paramsLmnn = {
    'lmnn__k': (1, 2, 4, 8, 16, 32),
    'lmnn__regularization': (.1, .5, .9),
    'lmnn__max_iter': defaultIters,
    'lmnn__learn_rate': (1e-7,)#, 1e-8, 1e-9),
}

paramsItml = {
#     'itml__num_constraints': (10, 100, 1000, 10000),
    'itml__gamma': (.01, .1, 1., 10.),
    'itml__max_iters': defaultIters,
}

paramsSdml = {
    'sdml__num_constraints': (10000, 100000),
    'sdml__use_cov': (True, False),
    'sdml__balance_param': (0.1, .25, .5, .75, 1),
    'sdml__sparsity_param': (.01, .05, .1, .25)
}

paramsLsml = {
#     'lsml__num_constraints': (100, 1000, 10000, 100000),
    'lsml__max_iter': defaultIters,
}

paramsNca = {
    'nca__max_iter': defaultIters,
    'nca__learn_rate': (0.1, 0.01),
}

paramsLfda = {
    'lfda__metric': ('weighted', 'orthonormalized'),
}

paramsRca = {
    'rca__num_chunks': (10, 50, 100, 500, 1000),
    'rca__chunk_size': (1, 2, 3, 5, 7, 10, 16, 32),
}

def pipelineKnn(X,y,datasetName):
    pipeline = [
        Pipeline([ ('imputer', defaultImputer), ]),
        Pipeline([ ('empty', EmptyTransformer()), ]),
        Pipeline([ ('knn', KNeighborsClassifier()), ]),
    ]
    params = [ {}, {}, defaultKnnParams, ]
    return evaluateClassifier(X, y, pipeline, params, "kNN", datasetName)

def pipelineCovKnn(X,y,datasetName):
    pipeline = [
        Pipeline([ ('imputer', defaultImputer), ]),
        Pipeline([ ('cov', Covariance()), ]),
        Pipeline([ ('knn', KNeighborsClassifier()), ]),
    ]
    params = [ {}, {}, defaultKnnParams, ]
    return evaluateClassifier(X, y, pipeline, params, "Cov+kNN", datasetName)

def pipelineCmaesKnn(X,y,datasetName):
    pipeline = [
        Pipeline([ ('imputer', defaultImputer), ]),
        Pipeline([ ('cmaes', MetricEvolution(strategy='cmaes', fitnesses='knn')) ]),
        Pipeline([ ('knn', KNeighborsClassifier()), ]),
    ]
    params = [ {}, paramsCmaes, defaultKnnParams, ]
    return evaluateClassifier(X, y, pipeline, params, "CMAES+kNN", datasetName)

def pipelineJdeKnn(X,y,datasetName):
    pipeline = [
        Pipeline([ ('imputer', defaultImputer), ]),
        Pipeline([ ('jde', MetricEvolution(strategy='jde', fitnesses='wfme')) ]),
        Pipeline([ ('knn', KNeighborsClassifier()), ]),
    ]
    params = [ {}, paramsJde, defaultKnnParams, ]
    return evaluateClassifier(X, y, pipeline, params, "JDE+kNN", datasetName)

def pipelineCmaesFmeKnn(X,y,datasetName):
    pipeline = [
        Pipeline([ ('imputer', defaultImputer), ]),
        Pipeline([ ('cmaesfme', MetricEvolution(strategy='cmaes', fitnesses='wfme')) ]),
        Pipeline([ ('knn', KNeighborsClassifier()), ]),
    ]
    params = [ {}, paramsCmaesFme, defaultKnnParams, ]
    return evaluateClassifier(X, y, pipeline, params, "CMAESFme+kNN", datasetName)

def pipelineJdePurKnn(X,y,datasetName):
    pipeline = [
        Pipeline([ ('imputer', defaultImputer), ]),
        Pipeline([ ('jdepur', MetricEvolution(strategy='jde', fitnesses='wpur')) ]),
        Pipeline([ ('knn', KNeighborsClassifier()), ]),
    ]
    params = [ {}, paramsJdePur, defaultKnnParams, ]
    return evaluateClassifier(X, y, pipeline, params, "JDEPur+kNN", datasetName)

def pipelineJdeKnnKnn(X,y,datasetName):
    pipeline = [
        Pipeline([ ('imputer', defaultImputer), ]),
        Pipeline([ ('jdeknn', MetricEvolution(strategy='jde', fitnesses='knn')) ]),
        Pipeline([ ('knn', KNeighborsClassifier()), ]),
    ]
    params = [ {}, paramsJdeKnn, defaultKnnParams, ]
    return evaluateClassifier(X, y, pipeline, params, "JDEkNN+kNN", datasetName)

def pipelineLmnnKnn(X,y,datasetName):
    pipeline = [
        Pipeline([ ('imputer', defaultImputer), ]),
        Pipeline([ ('lmnn', LMNN(k=3, min_iter=50, max_iter=1000, learn_rate=1e-07, regularization=0.5, convergence_tol=0.001)) ]),
        Pipeline([ ('knn', KNeighborsClassifier()), ]),
    ]
    params = [ {}, paramsLmnn, defaultKnnParams, ]
    return evaluateClassifier(X, y, pipeline, params, "LMNN+kNN", datasetName)


def pipelineItmlKnn(X,y,datasetName):
    pipeline = [
        Pipeline([ ('imputer', defaultImputer), ]),
        Pipeline([ ('itml', ITML_Supervised(gamma=1.,max_iters=1000,convergence_threshold=1e-3,num_constraints=None,bounds=None,A0=None)) ]),
        Pipeline([ ('knn', KNeighborsClassifier()), ]),
    ]
    params = [ {}, {**paramsItml, 'itml__num_constraints':noConstraints(y)}, defaultKnnParams, ]
    return evaluateClassifier(X, y, pipeline, params, "ITML+kNN", datasetName)


def pipelineSdmlKnn(X,y,datasetName):
    pipeline = [
        Pipeline([ ('imputer', defaultImputer), ]),
        Pipeline([ ('sdml', SDML_Supervised(balance_param=0.5, sparsity_param=0.01, use_cov=True, num_constraints=None)), ]),
        Pipeline([ ('knn', KNeighborsClassifier()), ]),
    ]
    params = [ {}, paramsSdml, defaultKnnParams, ]
    return evaluateClassifier(X, y, pipeline, params, "SDML+kNN", datasetName)


def pipelineLsmlKnn(X,y,datasetName):
    pipeline = [
        Pipeline([ ('imputer', defaultImputer), ]),
        Pipeline([ ('lsml', LSML_Supervised(tol=1e-3, max_iter=1000, prior=None, num_constraints=None)), ]),
        Pipeline([ ('knn', KNeighborsClassifier()), ]),
    ]
    params = [ {}, {**paramsLsml, 'lsml__num_constraints':noConstraints(y)}, defaultKnnParams, ]
    return evaluateClassifier(X, y, pipeline, params, "LSML+kNN", datasetName)


def pipelineNcaKnn(X,y,datasetName):
    pipeline = [
        Pipeline([ ('imputer', defaultImputer), ]),
        Pipeline([ ('nca', NCA(max_iter=100, learning_rate=0.01)), ]),
        Pipeline([ ('knn', KNeighborsClassifier()), ]),
    ]
    params = [ {}, paramsNca, defaultKnnParams, ]
    return evaluateClassifier(X, y, pipeline, params, "NCA+kNN", datasetName)


def pipelineLfdaKnn(X,y,datasetName):
    pipeline = [
        Pipeline([ ('imputer', defaultImputer), ]),
        Pipeline([ ('lfda', LFDA(num_dims=None, k=7, metric='weighted')), ]),
        Pipeline([ ('knn', KNeighborsClassifier()), ]),
    ]
    params = [ {}, {**paramsLfda, 'lfda__k': tuple(range(1, X.shape[1]))}, defaultKnnParams, ]
    return evaluateClassifier(X, y, pipeline, params, "LFDA+kNN", datasetName)


def pipelineRcaKnn(X,y,datasetName):
    pipeline = [
        Pipeline([ ('imputer', defaultImputer), ]),
        Pipeline([ ('rca', RCA_Supervised(dim=None, num_chunks=100, chunk_size=2)), ]),
        Pipeline([ ('knn', KNeighborsClassifier()), ]),
    ]
    params = [ {}, paramsRca, defaultKnnParams, ]
    return evaluateClassifier(X, y, pipeline, params, "RCA+kNN", datasetName)






def pipelineStandKnn(X,y,datasetName):
    pipeline = [
        Pipeline([
            ('imputer', defaultImputer),
            ('standardizer', defaultStandardizer),
        ]),
        Pipeline([ ('empty', EmptyTransformer()), ]),
        Pipeline([ ('knn', KNeighborsClassifier()), ]),
    ]
    params = [ {}, {}, defaultKnnParams, ]
    return evaluateClassifier(X, y, pipeline, params, "stand+kNN", datasetName)

def pipelineStandCovKnn(X,y,datasetName):
    pipeline = [
        Pipeline([
            ('imputer', defaultImputer),
            ('standardizer', defaultStandardizer),
        ]),
        Pipeline([ ('cov', Covariance()), ]),
        Pipeline([ ('knn', KNeighborsClassifier()), ]),
    ]
    params = [ {}, {}, defaultKnnParams, ]
    return evaluateClassifier(X, y, pipeline, params, "stand+Cov+kNN", datasetName)

def pipelineStandCmaesKnn(X,y,datasetName):
    pipeline = [
        Pipeline([
            ('imputer', defaultImputer),
            ('standardizer', defaultStandardizer),
        ]),
        Pipeline([ ('cmaes', MetricEvolution(strategy='cmaes', fitnesses='knn')) ]),
        Pipeline([ ('knn', KNeighborsClassifier()), ]),
    ]
    params = [ {}, paramsCmaes, defaultKnnParams, ]
    return evaluateClassifier(X, y, pipeline, params, "stand+CMAES+kNN", datasetName)

def pipelineStandCmaesFmeKnn(X,y,datasetName):
    pipeline = [
        Pipeline([
            ('imputer', defaultImputer),
            ('standardizer', defaultStandardizer),
        ]),
        Pipeline([ ('cmaesfme', MetricEvolution(strategy='cmaes', fitnesses='wfme')) ]),
        Pipeline([ ('knn', KNeighborsClassifier()), ]),
    ]
    params = [ {}, paramsCmaesFme, defaultKnnParams, ]
    return evaluateClassifier(X, y, pipeline, params, "stand+CMAESFme+kNN", datasetName)

def pipelineStandJdeKnn(X,y,datasetName):
    pipeline = [
        Pipeline([
            ('imputer', defaultImputer),
            ('standardizer', defaultStandardizer),
        ]),
        Pipeline([ ('jde', MetricEvolution(strategy='jde', fitnesses='wfme')) ]),
        Pipeline([ ('knn', KNeighborsClassifier()), ]),
    ]
    params = [ {}, paramsJde, defaultKnnParams, ]
    return evaluateClassifier(X, y, pipeline, params, "stand+JDE+kNN", datasetName)

def pipelineStandJdePurKnn(X,y,datasetName):
    pipeline = [
        Pipeline([
            ('imputer', defaultImputer),
            ('standardizer', defaultStandardizer),
        ]),
        Pipeline([ ('jdepur', MetricEvolution(strategy='jde', fitnesses='wpur')) ]),
        Pipeline([ ('knn', KNeighborsClassifier()), ]),
    ]
    params = [ {}, paramsJdePur, defaultKnnParams, ]
    return evaluateClassifier(X, y, pipeline, params, "stand+JDEPur+kNN", datasetName)

def pipelineStandJdeKnnKnn(X,y,datasetName):
    pipeline = [
        Pipeline([
            ('imputer', defaultImputer),
            ('standardizer', defaultStandardizer),
        ]),
        Pipeline([ ('jdeknn', MetricEvolution(strategy='jde', fitnesses='knn')) ]),
        Pipeline([ ('knn', KNeighborsClassifier()), ]),
    ]
    params = [ {}, paramsJdeKnn, defaultKnnParams, ]
    return evaluateClassifier(X, y, pipeline, params, "stand+JDEkNN+kNN", datasetName)

def pipelineStandLmnnKnn(X,y,datasetName):
    pipeline = [
        Pipeline([
            ('imputer', defaultImputer),
            ('standardizer', defaultStandardizer),
        ]),
        Pipeline([ ('lmnn', LMNN(k=3, min_iter=50, max_iter=1000, learn_rate=1e-07, regularization=0.5, convergence_tol=0.001)) ]),
        Pipeline([ ('knn', KNeighborsClassifier()), ]),
    ]
    params = [ {}, paramsLmnn, defaultKnnParams, ]
    return evaluateClassifier(X, y, pipeline, params, "stand+LMNN+kNN", datasetName)


def pipelineStandItmlKnn(X,y,datasetName):
    pipeline = [
        Pipeline([
            ('imputer', defaultImputer),
            ('standardizer', defaultStandardizer),
        ]),
        Pipeline([ ('itml', ITML_Supervised(gamma=1.,max_iters=1000,convergence_threshold=1e-3,num_constraints=None,bounds=None,A0=None)) ]),
        Pipeline([ ('knn', KNeighborsClassifier()), ]),
    ]
    params = [ {}, {**paramsItml, 'itml__num_constraints':noConstraints(y)}, defaultKnnParams, ]
    return evaluateClassifier(X, y, pipeline, params, "stand+ITML+kNN", datasetName)


def pipelineStandSdmlKnn(X,y,datasetName):
    pipeline = [
        Pipeline([
            ('imputer', defaultImputer),
            ('standardizer', defaultStandardizer),
        ]),
        Pipeline([ ('sdml', SDML_Supervised(balance_param=0.5, sparsity_param=0.01, use_cov=True, num_constraints=None)), ]),
        Pipeline([ ('knn', KNeighborsClassifier()), ]),
    ]
    params = [ {}, paramsSdml, defaultKnnParams, ]
    return evaluateClassifier(X, y, pipeline, params, "stand+SDML+kNN", datasetName)


def pipelineStandLsmlKnn(X,y,datasetName):
    pipeline = [
        Pipeline([
            ('imputer', defaultImputer),
            ('standardizer', defaultStandardizer),
        ]),
        Pipeline([ ('lsml', LSML_Supervised(tol=1e-3, max_iter=1000, prior=None, num_constraints=None, verbose=False)), ]),
        Pipeline([ ('knn', KNeighborsClassifier()), ]),
    ]
    params = [ {}, {**paramsLsml, 'lsml__num_constraints':noConstraints(y)}, defaultKnnParams, ]
    return evaluateClassifier(X, y, pipeline, params, "stand+LSML+kNN", datasetName)


def pipelineStandNcaKnn(X,y,datasetName):
    pipeline = [
        Pipeline([
            ('imputer', defaultImputer),
            ('standardizer', defaultStandardizer),
        ]),
        Pipeline([ ('nca', NCA(max_iter=100, learning_rate=0.01)), ]),
        Pipeline([ ('knn', KNeighborsClassifier()), ]),
    ]
    params = [ {}, paramsNca, defaultKnnParams, ]
    return evaluateClassifier(X, y, pipeline, params, "stand+NCA+kNN", datasetName)


def pipelineStandLfdaKnn(X,y,datasetName):
    pipeline = [
        Pipeline([
            ('imputer', defaultImputer),
            ('standardizer', defaultStandardizer),
        ]),
        Pipeline([ ('lfda', LFDA(num_dims=None, k=7, metric='weighted')), ]),
        Pipeline([ ('knn', KNeighborsClassifier()), ]),
    ]
    params = [ {}, {**paramsLfda, 'lfda__k': tuple(range(1, X.shape[1]))}, defaultKnnParams, ]
    return evaluateClassifier(X, y, pipeline, params, "stand+LFDA+kNN", datasetName)


def pipelineStandRcaKnn(X,y,datasetName):
    pipeline = [
        Pipeline([
            ('imputer', defaultImputer),
            ('standardizer', defaultStandardizer),
        ]),
        Pipeline([ ('rca', RCA_Supervised(dim=None, num_chunks=100, chunk_size=2)), ]),
        Pipeline([ ('knn', KNeighborsClassifier()), ]),
    ]
    params = [ {}, paramsRca, defaultKnnParams, ]
    return evaluateClassifier(X, y, pipeline, params, "stand+RCA+kNN", datasetName)


# In[ ]:

import glob, os

datasets = []
for file in glob.glob("{}/*.csv".format(datasetsDirectory)):
    datasets.append(file)
datasets.sort()

# datasets.remove('datasets/soybean-large.csv')

datasets = [
'datasets/balance-scale.csv',
'datasets/breast-cancer.csv',
'datasets/gaussians.csv',
'datasets/iris.csv',
'datasets/pima-indians.csv',
'datasets/wine.csv',

# 'datasets/ionosphere.csv',
'datasets/mice-protein.csv',
'datasets/sonar.csv',
# 'datasets/soybean-large.csv',

'datasets/digits6.csv',
'datasets/digits10.csv',
]

# datasets = datasets[4:5]
logging.info("Datasets: " + str(datasets))

for x in datasets:
    print(x, pd.read_csv(x, sep=',', skiprows=1, header=0).shape)


# In[ ]:

for filename in datasets:
    results = []
    datasetName = filename[len(datasetsDirectory)+1:-4]
    
    logging.info("Starting `{}` dataset".format(datasetName))

    data = pd.read_csv(filename, sep=',', skiprows=1, header=0)

    y = data['class']
    X = data.drop(['class'], axis=1).values
    
    le = LabelEncoder()
    y = le.fit_transform(y)
    
#     known_label_idx, = np.where(y >= 0)
#     known_labels = y[known_label_idx]
#     uniq, lookup = np.unique(known_labels, return_inverse=True)
#     all_inds = [set(np.where(lookup==c)[0]) for c in range(len(uniq))]
#     print(len(all_inds)-1)
#     print(np.random.randint(0, high=len(all_inds)-1))

#     evaluatePipeline( X,y,datasetName,pipelineKnn )
#     evaluatePipeline( X,y,datasetName,pipelineCovKnn )
#     evaluatePipeline( X,y,datasetName,pipelineCmaesKnn )
#     evaluatePipeline( X,y,datasetName,pipelineCmaesFmeKnn )
#     evaluatePipeline( X,y,datasetName,pipelineJdeKnn )
#     evaluatePipeline( X,y,datasetName,pipelineJdePurKnn )
#     evaluatePipeline( X,y,datasetName,pipelineJdeKnnKnn )
# #     evaluatePipeline( X,y,datasetName,pipelineLmnnKnn )
#     evaluatePipeline( X,y,datasetName,pipelineItmlKnn )
#     evaluatePipeline( X,y,datasetName,pipelineSdmlKnn )
#     evaluatePipeline( X,y,datasetName,pipelineLsmlKnn )
# #     evaluatePipeline( X,y,datasetName,pipelineNcaKnn )
    evaluatePipeline( X,y,datasetName,pipelineLfdaKnn )
#     evaluatePipeline( X,y,datasetName,pipelineRcaKnn ) 

#     evaluatePipeline( X,y,datasetName,pipelineStandKnn )
#     evaluatePipeline( X,y,datasetName,pipelineStandCovKnn )
#     evaluatePipeline( X,y,datasetName,pipelineStandCmaesKnn )
#     evaluatePipeline( X,y,datasetName,pipelineStandCmaesFmeKnn )
#     evaluatePipeline( X,y,datasetName,pipelineStandJdeKnn )
#     evaluatePipeline( X,y,datasetName,pipelineStandJdePurKnn )
#     evaluatePipeline( X,y,datasetName,pipelineStandJdeKnnKnn )
# #     evaluatePipeline( X,y,datasetName,pipelineStandLmnnKnn )
#     evaluatePipeline( X,y,datasetName,pipelineStandItmlKnn )
#     evaluatePipeline( X,y,datasetName,pipelineStandSdmlKnn )
#     evaluatePipeline( X,y,datasetName,pipelineStandLsmlKnn )
# #     evaluatePipeline( X,y,datasetName,pipelineStandNcaKnn )
    evaluatePipeline( X,y,datasetName,pipelineStandLfdaKnn )
#     evaluatePipeline( X,y,datasetName,pipelineStandRcaKnn ) 


# In[ ]:




# In[ ]:

# with open('{}/{}'.format(dumpsDirectory, 'dump_Cov+kNN_balance-scale.bin'), 'rb') as input:
#     h = pickle.load(input)

# print(h.steps[-2][-1][0].steps[0][1].M)



# In[ ]:

# ## RENAMING files to shorter
# results = []
# for file in glob.glob("{}/*.csv".format(resultsDirectory)):
#     results.append(file)
# results.sort()

# for x in results:
#     y = x.replace('breast-cancer-wisconsin', 'breast-cancer')
#     y = y.replace('pima-indians-diabetes', 'pima-indians')
    
#     os.rename(x, y)


# In[ ]:

# def inplace_change(filename):
#     # Safely read the input filename using 'with'
#     with open(filename) as f:
#         s = f.read()
#         if ('breast-cancer-wisconsin' not in s) and ('pima-indians-diabetes' not in s):
#             print('not found in {}'.format(filename))
#             return

#     # Safely write the changed content, if found in the file
#     with open(filename, 'w') as f:
#         s = s.replace('breast-cancer-wisconsin', 'breast-cancer')
#         s = s.replace('pima-indians-diabetes', 'pima-indians')
#         f.write(s)
        
# # ## REPLACING files to shorter in index
# results = []
# for file in glob.glob("{}/*.csv".format(resultsDirectory)):
#     results.append(file)
# results.sort()

# for x in results:
#     inplace_change(x)

