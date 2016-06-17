import numpy as np
from sklearn.externals.joblib import Parallel, delayed
from sklearn.grid_search import ParameterGrid
from sklearn.base import clone
from time import time
import logging
import json

class HierarchicalGridSearchCV(object):
    def __init__(self, estimators, params, cv=None,
                 scoring=None, verbose=False, iid=True,
                 n_jobs=1, pre_dispatch='2*n_jobs'):
        self.estimators = estimators
        self.params = params
        self.cv = cv
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.pre_dispatch = pre_dispatch
        self.iid = iid
        
        self.n_levels = len(estimators)
        self.steps = [None] * (self.n_levels+1)
        
        if len(estimators) != len(params):
            raise ValueError("Estimators and params have different length!")
            
            
    def estimator_fit_transform(self,estimator,parameters,old_parameters,
                                X_train,y_train,X_test,y_test,times,error,
                                transform=True):
        newParamDict = {**old_parameters, **parameters}

        if error is not None:
            return None,None,None,None,y_test,newParamDict,times+(0,),"Cascading:{}".format(error)

        estimator.set_params(**parameters)
        t0 = time()
        try:
            if transform:
                X_train_tr = estimator.fit_transform(X_train, y_train)
                return estimator,X_train_tr,y_train,estimator.transform(X_test),y_test,newParamDict,times+(time()-t0,),error
            else:
                estimator.fit(X_train, y_train)
                return estimator,None,None,X_test,y_test,newParamDict,times+(time()-t0,),error
        except Exception as e:
            logging.info("\nError in HirearchicalGridSearch pipeline")
            logging.info( json.dumps(newParamDict, sort_keys=True, indent=4))
            logging.exception("Exception in step: {}".format(",".join([name for name,e in estimator.steps])))
            return None,None,None,None,y_test,newParamDict,times+(time()-t0,),str(e)
            
    def fit(self, X, y):
        n_folds = self.cv.n_folds
        
        # init step: only CV
        self.steps[0] = [ (None,X[train],y[train],X[test],y[test],{},tuple(),None) for train,test in self.cv ]
        
        for step in range(self.n_levels):
            parameter_iterable = ParameterGrid(self.params[step])
            last_estimator = step+1==self.n_levels

            n_jobs = 1 if last_estimator else self.n_jobs # only 1 job for kNN

            self.steps[step+1] = Parallel(
                n_jobs=n_jobs, verbose=self.verbose,
                pre_dispatch=self.pre_dispatch
            )(
                delayed(self.estimator_fit_transform, check_pickle=False)(
                    clone(self.estimators[step]), parameters, old_parameters,
                    X_train, y_train, X_test, y_test, times, err, transform=(not last_estimator)
                )
                    for parameters in parameter_iterable
                    for est,X_train,y_train,X_test,y_test,old_parameters,times,err in self.steps[step]
            )
          
        self.scores = [ (est.score(X_test, y_test),len(y_test),parameters,times,err) if est is not None else (0,len(y_test),parameters,times,err)
                       for est,X_train,y_train,X_test,y_test,parameters,times,err in self.steps[-1] ]
        n_fits = len(self.scores)
        
        # FROM https://github.com/scikit-learn/scikit-learn/blob/51a765a/sklearn/grid_search.py#L560
        self.grid_scores_ = list()
        non_empty_error = None
        times_sum = None
        for grid_start in range(0, n_fits, n_folds):
            n_test_samples = 0
            score = 0
            all_scores = []
            for this_score, this_n_test_samples, parameters, times, err in self.scores[grid_start:grid_start + n_folds]:
                times_sum = times if times_sum is None else [t1+t2 for t1,t2 in zip(times_sum, times)]
                non_empty_error = err if err is not None else non_empty_error
                all_scores.append(this_score)
                if self.iid:
                    this_score *= this_n_test_samples
                    n_test_samples += this_n_test_samples
                score += this_score
            if self.iid:
                score /= float(n_test_samples)
            else:
                score /= float(n_folds)

            # make output statistics
            times_dict = {}
            for i,t in enumerate([float(x)/n_folds for x in times_sum]):
                times_dict['time'+repr(i+1)] = t

            scores_dict = {}
            for i,t in enumerate(all_scores):
                scores_dict['score'+repr(i+1)] = t

            self.grid_scores_.append({
                'mean': score if non_empty_error is None else 0,
                'std': np.std(all_scores),
                'params': parameters,
                'scores': scores_dict,
                'times': times_dict,
                'error': non_empty_error,
            })

        return self
