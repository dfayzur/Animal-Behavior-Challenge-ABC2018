import numpy as np
import pandas as pd
import gc

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import KFold, StratifiedKFold

import warnings

from sklearn import metrics


import ml_metrics
np.random.RandomState(2018)
np.random.seed(2018)
import random
random.seed(2018)

warnings.simplefilter(action='ignore', category=FutureWarning)

np.random.randint(5000, 10000, size=10).tolist()
random.sample(range(5000, 10000), 10)


def stacking_model_sk_et(task='together'):
    nfold = 5
    #task='together'

    train_df = None
    test_df = None

    if task == 'together':
        train_df = pd.read_csv('./data/train_df_day_night_together.csv')
        test_df = pd.read_csv('./data/test_df_day_night_together.csv')
        from together_fn_param import list_param
    elif task == 'split':
        train_df = pd.read_csv('./data/train_df_day_night_split.csv')
        test_df = pd.read_csv('./data/test_df_day_night_split.csv')
        from split_fn_param import list_param

    train_df = train_df.fillna(-1)
    test_df = test_df.fillna(-1)

    print("Data loading Done!")
    target = 'label'
    predictors = train_df.columns.values.tolist()[1:-1]
    categorical = None

    X_train = train_df[predictors].values
    X_test = test_df[predictors].values
    labels = train_df['label']

    seeds = np.random.randint(5000, 10000, size=10).tolist()
    auc_lst = []
    auc_lst1 = []
    n_estimators_lst =[]
    stratified=True
    debug = True
    param = list_param('sk_et')
    oof_preds_folds = np.zeros((train_df.shape[0],len(seeds)))
    sub_preds_folds = np.zeros((test_df.shape[0],len(seeds)))
    sub_preds_folds_vote = np.zeros((test_df.shape[0],len(seeds)))
    oof_preds_folds_vote = np.zeros((train_df.shape[0],len(seeds)))
    feature_importance_df_folds = pd.DataFrame()

    list_thresholds_global = []
    for seed_id in range(len(seeds)):
        if stratified:
            folds = StratifiedKFold(n_splits=nfold, shuffle=True, random_state=seeds[seed_id])
        else:
            folds = KFold(n_splits=nfold, shuffle=True, random_state=1001)
        oof_preds = np.zeros(train_df.shape[0])
        sub_preds = np.zeros(test_df.shape[0])
        oof_preds_local_vote = np.zeros(train_df.shape[0])
        sub_preds_local_vote = np.zeros((test_df.shape[0], nfold))
        feature_importance_df = pd.DataFrame()

        gfold_Id = list(folds.split(X_train, labels))
        params_iter = {
            'random_state':seeds[seed_id]
        }
        param.update(params_iter)

        clf = ExtraTreesClassifier(
            n_estimators=int(param['n_estimators']),
            min_samples_split=int(param['min_samples_split']),
            min_samples_leaf=int(param['min_samples_leaf']),
            max_depth=int(param['max_depth']),
            max_features=param['max_features'],
            #max_leaf_nodes=param['max_leaf_nodes'],
            random_state=seeds[seed_id],
            bootstrap=True,
            oob_score = True,
            n_jobs=-1)

        for n_fold, (train_idx, valid_idx) in enumerate(folds.split(X_train, labels)):
            xtrain, xtest = X_train[train_idx, :], X_train[valid_idx, :]
            ytrain, ytest = labels[train_idx], labels[valid_idx]

            clf.fit(xtrain, ytrain)

            oof_preds[valid_idx] = clf.predict_proba(xtest)[:, 1]
            pred = clf.predict_proba(X_test)[:, 1]
            sub_preds += pred / folds.n_splits

            fpr, tpr, thresholds = metrics.roc_curve(train_df[target].iloc[valid_idx], oof_preds[valid_idx])
            optimal_idx = np.argmax(tpr - fpr)
            optimal_thresholds = thresholds[optimal_idx]

            list_thresholds_global.append(optimal_thresholds)

            sub_preds_local_vote[:, n_fold] = [1 if y_cont > optimal_thresholds else 0 for y_cont in pred]
            oof_preds_local_vote[valid_idx] = [1 if y_cont > optimal_thresholds else 0 for y_cont in oof_preds[valid_idx]]


            print('Fold %2d AUC : %.6f' % (n_fold + 1, roc_auc_score(ytest, oof_preds[valid_idx])))

            del xtrain, xtest, ytrain, ytest
            gc.collect()

        oof_preds_folds[:, seed_id]=oof_preds
        sub_preds_folds[:, seed_id] = sub_preds
        from scipy import stats

        a, b = stats.mode(sub_preds_local_vote, axis=1)
        oof_preds_folds_vote[:, seed_id] = oof_preds_local_vote
        sub_preds_folds_vote[:, seed_id] = a.reshape(-1)
        feature_importance_df_folds = pd.concat([feature_importance_df_folds, feature_importance_df], axis=0)
        auc_lst.append(ml_metrics.auc(train_df[target], oof_preds))
        auc_lst1.append(roc_auc_score(train_df[target], oof_preds))
        print('Full AUC score %.6f' % roc_auc_score(train_df[target], oof_preds))
        print("auc_lst1")
        print(auc_lst1)


    print(list_thresholds_global)
    #oof_preds_folds = pd.DataFrame(oof_preds_folds,columns=['sk_et_seed_' + str(seeds[l]) for l in range(len(seeds))])
    #sub_preds_folds = pd.DataFrame(sub_preds_folds,columns=['sk_et_seed_' + str(seeds[l]) for l in range(len(seeds))])
    oof_preds_folds_vote = pd.DataFrame(oof_preds_folds_vote,columns=['sk_et_seed_' + str(seeds[l]) for l in range(len(seeds))])
    sub_preds_folds_vote = pd.DataFrame(sub_preds_folds_vote,columns=['sk_et_seed_' + str(seeds[l]) for l in range(len(seeds))])

    #oof_preds_folds.to_csv("./output/" + task + "_train_stack/sk_et.csv", index=False)
    #sub_preds_folds.to_csv("./output/" + task + "_test_stack/sk_et.csv", index=False)
    oof_preds_folds_vote.to_csv("./output/" + task + "_train_stack_vote/sk_et.csv", index=False)
    sub_preds_folds_vote.to_csv("./output/" + task + "_test_stack_vote/sk_et.csv", index=False)

