import argparse
import json
import os
import random
import sys
import numpy as np
import pandas as pd

from joblib import dump
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectPercentile, chi2
from sklearn.pipeline import Pipeline


label_names = ['political',
                'military',
                'social',
                'economic',
                'infrastructure',
                'information',
                'economic_production',
                'institutions',
                'leadership',
                'transportation',
                'economic_distribution',
                'economic_consumption',
                'economic_informality',
                'global_info',
                'affinity_groups',
                'political_org',
                'central_gov',
                'military_ind_base',
                'climate',
                'weather',
                'sustainment',
                'utilities',
                'national_info',
                'armed_forces',
                'habitability',
                'military_short',
                'economic_performance',
                'internal_security',
                'basic_needs',
                'defense_infrastructure',
                'phys_environment']


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def get_indices_at_thresh(y_hat, thresh:float, return_data:bool=False):
    ix = []
    for row in range(y_hat.shape[0]):
        if any([i >= thresh for i in y_hat[row, :]]):
            ix.append(row)

    print(len(ix))

    if return_data:
        return ix


def fit_initial_model():
    df = pd.read_json('/data/users/kyle.shaffer/proj-wm-data/small_train.jl', lines=True)
    # Filter down to non-empty cases
    df = df[df.content.notnull()]
    df = df[df.content != '']

    X = df.content.apply(lambda x: x.strip().replace('\n', ' ')).values # np.load('/data/users/kyle.shaffer/proj-wm-data/small_data_vecs.npy')
    y = df[label_names].values.astype(int) # df.int_label.values

    # Quick train-test split
    ix = list(range(len(y)))
    random.shuffle(ix)
    train_ix = ix[:int(0.75 * len(y))]
    test_ix = [i for i in ix if not(i in set(train_ix))]
    X_train = X[train_ix]
    X_test = X[test_ix]
    y_train = y[train_ix, :]
    y_test = y[test_ix, :]
    print('TRAIN EXAMPLES: {}\tTEST EXAMPLES: {}'.format(len(X_train), len(X_test)))

    model = OneVsRestClassifier(LinearSVC(max_iter=5000))
    pipe = Pipeline(
        [
            ('tfidf', TfidfVectorizer()),
            ('selector', SelectPercentile(score_func=chi2)),
            ('estimator', model)
        ], verbose=True
    )
    print('Running pipeline model')
    print(pipe)

    param_grid = {'estimator__estimator__C': [0.1, 0.5, 1, 3, 5], 'estimator__estimator__class_weight': [None, 'balanced']}
    grid = GridSearchCV(estimator=pipe, param_grid=param_grid, refit=True, scoring='f1_macro', cv=3, n_jobs=2, verbose=1)

    print('Training classifier...')
    grid.fit(X_train, y_train)
    print('Getting predictions')
    y_pred = grid.predict(X_test)

    print('Classification Results')
    print(metrics.classification_report(y_test, y_pred, target_names=label_names, digits=4))

    # Fit on entire dataset
    grid.fit(X, y)
    print(metrics.classification_report(grid.predict(X), y, target_names=label_names, digits=4))

    return grid, X, y


def sample_from_model(model, X_lab, y_lab, X_nolab, thresh, min_samples, max_iterations):
    # Get initial new labels from pre-trained model
    y_hat = model.predict(X_nolab)
    y_prob = sigmoid(model.decision_function(X_nolab))

    ix = get_indices_at_thresh(y_hat=y_prob, thresh=thresh, return_data=True)
    ix_unlabelled = [i for i in range(y_hat.shape[0]) if not(i in set(ix))]

    X_samp = X_nolab[ix]
    print('X_samp:', X_samp.shape)
    y_samp = y_hat[ix]
    X_train = np.concatenate([X_lab, X_samp])
    print('X_train:', len(X_train))
    y_train = np.concatenate([y_lab, y_samp])
    X_nolab = X_nolab[ix_unlabelled]

    fit = True
    iteration = 1
    while fit:
        print()
        print('\nRunning iteration {}...\n'.format(iteration))
        model.best_estimator_.fit(np.asarray(X_train), y_train)
        y_hat = model.best_estimator_.predict(X_nolab)
        y_prob = sigmoid(model.best_estimator_.decision_function(X_nolab))
        ix = get_indices_at_thresh(y_hat=y_prob, thresh=thresh, return_data=True)

        if (len(ix) < min_samples) or (iteration > max_iterations):
            fit = False

        ix_unlabelled = [i for i in range(y_hat.shape[0]) if not(i in set(ix))]

        X_samp = X_nolab[ix].tolist()
        y_samp = y_hat[ix]

        X_train = np.concatenate([X_train, X_samp])
        y_train = np.concatenate([y_train, y_samp])
        X_nolab = X_nolab[ix_unlabelled]

        print('\n\nNew number of training samples: {}\n\n'.format(len(X_train)))

        iteration += 1

    dump(model.best_estimator_, 'expanded_cat_model_tfidf.joblib')
    print('Model saved!')

    np.save('/data/users/kyle.shaffer/proj-wm-data/semi_sup_cats_tfidf_data.npy', X_train)
    np.save('/data/users/kyle.shaffer/proj-wm-data/semi_sup_cats_labs_tfidf_model.npy', y_train)
    print('Data saved!')
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample_thresh', type=float, required=False, default=0.7)
    parser.add_argument('--min_samples', type=int, required=False, default=40)
    parser.add_argument('--max_iterations', type=int, required=False, default=100)
    args = parser.parse_args()

    init_model, X_lab, y_lab = fit_initial_model()
    # unlabelled_data = np.load('/data/users/kyle.shaffer/proj-wm-data/baltics8k_vecs.npy')
    unlabelled_data_dir = '/data/users/kyle.shaffer/proj-wm-data/baltics-8k-sub-corpus'
    unlabelled_data = []
    for fname in os.listdir(unlabelled_data_dir):
        sys.stdout.write('\rReading file: {}...'.format(fname))
        if fname.endswith('json'):
            with open(os.path.join(unlabelled_data_dir, fname), mode='r') as infile:
                json_data = json.load(infile)
                if 'extracted_text' in json_data.keys():
                    text = json_data['extracted_text'].replace('\n', ' ')
                    if (len(text) > 0) and (text != '\n') and (text != ' '):
                        unlabelled_data.append(text)

    print('\nUnlabelled data read in...')

    sample_from_model(init_model, X_lab, y_lab, np.asarray(unlabelled_data),
                      args.sample_thresh, args.min_samples, args.max_iterations)