#!/usr/bin/python3


from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier
)

from sklearn.linear_model import (
    LogisticRegression,
    SGDClassifier,
    SGDOneClassSVM,
    Perceptron
)

from sklearn.svm import OneClassSVM, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
#from xgboost import XGBClassifier

from sklearn.multiclass import OneVsRestClassifier
from kfold import FewShotKFold
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
import numpy as np
import pandas as pd
import pickle
import argparse
import ray
import logging
import time
import os
import yaml

import loader


def profile(fn):
    def wrapper(*args, **kwds):
        begin = time.time()
        res = fn(*args, **kwds)
        print("Time", time.time() - begin)
        return res
    return wrapper


def ray_session(fn):
    def wrapper(*args, **kwds):
        ray.init()
        res = fn(*args, **kwds)
        ray.shutdown()
        return res
    return wrapper


MODELS = dict(
    OneClassSVM=OneClassSVM,
    GradBoost=GradientBoostingClassifier,
    LogisticRegression=LogisticRegression,
    DecisionTree=DecisionTreeClassifier,
    RandomForest=RandomForestClassifier,
    SVM=SVC,
    KNN=KNeighborsClassifier,
#    XGBoost=XGBClassifier,
    GaussianNB=GaussianNB,
    AdaBoost=AdaBoostClassifier,
    Perceptron=Perceptron,
    SGD=SGDClassifier,
    SGDOneClassSVM=SGDOneClassSVM,
)


def model_loader(modelname: str,
                 multiclass: bool = False,
                 cross_valid: int = None,
                 grid_search: dict = None,
                 model_params: dict = {}):
    if modelname == 'OneClassSVM':
        model_params.setdefault('kernel', 'rbf')
        model_params.setdefault('gamma', 5e-5)

    model = MODELS[modelname](**model_params)

    if multiclass:
        model = OneVsRestClassifier(model)

    if grid_search is not None:
        model = GridSearchCV(model, grid_search, cv=cross_valid,
                             n_jobs=-1, scoring="accuracy", verbose=10)

    return model


def dataset_loader(dataset,
                   n_mal: int,
                   n_ben: int,
                   verbose: bool = False,
                   multiclass: bool = False):
    if n_ben is None:
        n_ben = len(dataset['benign_y']) + 1
    if n_mal is None:
        n_mal = len(dataset['malicious_y']) + 1

    attack_label = dataset['malicious_y'][:n_mal]
    attack_data = dataset['malicious_x'][:n_mal]
    normal_label = dataset['benign_y'][:n_ben]
    normal_data = dataset['benign_x'][:n_ben]

    if not multiclass:
        normal_label[normal_label != 0] = +1
        normal_label[normal_label == 0] = -1
        attack_label[attack_label != 0] = +1
        attack_label[attack_label == 0] = -1

    if verbose:
        print({k: tuple(attack_label).count(k)
               for k in np.unique(attack_label)})
        print('train has', normal_label.shape[0], 'normal',
              'and', attack_label.shape[0], 'attack samples')

    X = np.concatenate([normal_data, attack_data])
    y = np.concatenate([normal_label, attack_label])

    return X, y


@ray_session
def parallel_cv_main(X, y, modelname, cv, **model_params):
    for fold_idx, (train_index, test_index) in enumerate(cv.split(X, y)):
        train_data, train_label = X[train_index], y[train_index]
        test_data, test_label = X[test_index], y[test_index]
        yield parallel_cv.remote(modelname, model_params,
                                 train_data, train_label,
                                 test_data, test_label)


@ray.remote
def parallel_cv(modelname, model_params,
                X, y, X_test, y_test, average='binary'):
    model = model_loader(modelname, model_params=model_params)
    model.fit(X, y)
    y_pred = model.predict(X_test)
    cv = metrics.confusion_matrix(y_test, y_pred, labels=[-1, 1])
    return dict(accuracy=metrics.accuracy_score(y_test, y_pred),
                precision=metrics.precision_score(y_test, y_pred, average=average),
                recall=metrics.recall_score(y_test, y_pred, average=average),
                f1=metrics.f1_score(y_test, y_pred, average=average),
                n_mal_in_train=sum(y > 0),
                n_ben_in_train=sum(y <= 0),
                n_mal_in_test=len(y_test),
                num_tn=cv[1, 1],
                num_fn=cv[1, 0])


@profile
def best_model_search(dataset, modelname: str, grid_search,
                      multiclass=False, n_splits: int = 10,
                      n_benigns: int = None, n_malicious: int = None):
    X, y = dataset_loader(dataset,
                          n_ben=n_benigns,
                          n_mal=n_malicious,
                          multiclass=multiclass)
    if modelname == 'XGBoost' and not multiclass:
        y[y == -1] = 0

    print("The dataset has",
          sum(y > 0), "malicious",
          sum(y <= 0), "benign traffic")

    cv = FewShotKFold(target_class=1, n_splits=n_splits,
                      shuffle=True, random_state=42)

    model = model_loader(modelname=modelname,
                         cross_valid=cv,
                         multiclass=multiclass,
                         grid_search=grid_search)
    model.fit(X, y)
    print("best conf", model.best_params_)
    print(f"best score {model.best_score_:.4f}%")
    return model.best_estimator_, model.best_params_


@profile
def do_experiment(dataset, modelname: str, model_params: dict,
                  multiclass=False, n_splits: int = 10,
                  n_benigns: int = None, n_malicious: int = None):
    X, y = dataset_loader(dataset,
                          n_ben=n_benigns,
                          n_mal=n_malicious,
                          multiclass=multiclass)
    if modelname == 'XGBoost' and not multiclass:
        y[y == -1] = 0

    cv = FewShotKFold(target_class=1, n_splits=n_splits,
                      shuffle=True, random_state=42)
    fold_results = parallel_cv_main(modelname=modelname,
                                    X=X, y=y, cv=cv, **model_params)
    return pd.DataFrame(ray.get(list(fold_results)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=MODELS.keys(), required=True)
    parser.add_argument('--prefix', type=str, required=True)
    parser.add_argument('--n-ben', type=int, default=None)
    parser.add_argument('--n-mal', type=int, default=None)
    parser.add_argument('--multiclass', default=False, action='store_true')
    parser.add_argument('--grid-search', default=False, action='store_true')
    train_dataset, test_dataset = loader.load_train_test_datasets(patient='0005', task='classification', input_type='power-spectral-coeff')
    from sklearn.metrics import accuracy_score
    #params = parser.parse_args()

    model = RandomForestClassifier(n_estimators=48)
    linear_coeffs = np.concatenate(train_dataset.args['linear_coeffs'])
    quadratic_coeffs = np.concatenate(train_dataset.args['quadratic_coeffs'])
    X = np.concatenate([linear_coeffs, quadratic_coeffs], 1)
    y = np.concatenate(train_dataset.args['observation'])
    y[np.where(y == 0)] = -1

    model.fit(X, y)

    for key, value in test_dataset.items():
        linear_coeffs = np.concatenate(value.args['linear_coeffs'])
        quadratic_coeffs = np.concatenate(value.args['quadratic_coeffs'])
        X = np.concatenate([linear_coeffs, quadratic_coeffs], 1)
        y = np.concatenate(value.args['observation'])
        y[np.where(y == 0)] = -1
        y_pred = model.predict(X)
        acc = accuracy_score(y_pred=y_pred, y_true=y)
        print(key, acc)
        import matplotlib.pyplot as plt
        plt.plot(y, label=f'{key} ground truth')
        plt.plot(y_pred, label=f'{key} predictions')
        plt.savefig(key + '.png')
        plt.cla()
        #import pdb; pdb.set_trace()

    1/0
    dataset = np.load('dataset.npz')
    profilers = dict()

    with open('param_search.yaml', 'r') as f:
        param_search = yaml.safe_load(f)

    os.makedirs(params.prefix, exist_ok=True)
    if params.grid_search:
        assert params.model in param_search
        # os.sys.stdout = open(os.path.join(params.prefix, 'stdout.log'), 'w')
        # os.sys.stderr = open(os.path.join(params.prefix, 'stderr.log'), 'w')
        logging.basicConfig(filename=os.path.join(params.prefix, 'grid_search.log'),
                            format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                            datefmt='%H:%M:%S', level=10)

        profilers['best_model_search'] = -time.time()
        best_model, model_conf = best_model_search(
                modelname=params.model, dataset=dataset,
                grid_search=param_search[params.model],
                multiclass=params.multiclass, n_splits=10,
                n_benigns=params.n_ben, n_malicious=params.n_mal)
        profilers['best_model_search'] += time.time()
        with open(os.path.join(params.prefix, 'best-model.pk'), 'wb') as f:
            pickle.dump(best_model, f)
    else:
        model_conf = {}

    profilers['evaluation'] = -time.time()
    results = do_experiment(dataset=dataset,
                            modelname=params.model,
                            model_params=model_conf,
                            n_benigns=params.n_ben,
                            n_malicious=params.n_mal,
                            multiclass=params.multiclass)
    profilers['evaluation'] += time.time()
    print(results.mean(0))
    results.to_csv(os.path.join(params.prefix, 'k-fold-results.csv'))
    if params.grid_search:
        with open(os.path.join(params.prefix, 'best-params.yaml'), 'w') as f:
            yaml.dump({'model': dict(name=best_model.__class__.__name__,
                                     params=model_conf,
                                     scores=results.mean(0).to_dict(),
                                     profile=profilers)}, f)
