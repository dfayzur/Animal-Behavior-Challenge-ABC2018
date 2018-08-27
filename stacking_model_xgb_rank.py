# Forked from excellent kernel : https://www.kaggle.com/jsaguiar/updated-0-792-lb-lightgbm-with-simple-features
# From Kaggler : https://www.kaggle.com/jsaguiar
# Just added a few features so I thought I had to make release it as well...

import numpy as np
import pandas as pd
import gc
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import KFold, StratifiedKFold

import warnings
from sklearn.metrics import confusion_matrix
from sklearn import metrics

import xgboost as xgb
import ml_metrics
np.random.RandomState(2018)
np.random.seed(2018)
import random
random.seed(2018)

warnings.simplefilter(action='ignore', category=FutureWarning)

np.random.randint(5000, 10000, size=10).tolist()
random.sample(range(5000, 10000), 10)


def stacking_model_xgb_rank(task='together'):
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

    X_train = train_df.drop(['bird_id', 'label'], axis=1)
    features = X_train.columns
    labels = train_df['label']
    X_train = X_train.fillna(-1)
    y_train = np.int32(labels)

    X_test = test_df.drop(['bird_id', 'label'], axis=1)
    X_test = X_test.fillna(-1)

    #xgboost

    xg_train = xgb.DMatrix(X_train, label=y_train, missing=-1.0)
    xg_test = xgb.DMatrix(X_test, missing=-1.0)

    def xg_f1(yhat,dtrain):
        y = dtrain.get_label()

        pre, rec, th = metrics.precision_recall_curve(y, yhat)

        f1_all = 2 / ((1 / rec) + (1 / pre))
        optimal_idx = np.argmax(f1_all)
        optimal_thresholds = th[optimal_idx]
        y_bin = [1. if y_cont > optimal_thresholds else 0. for y_cont in yhat]  # binaryzing your output
        tn, fp, fn, tp = confusion_matrix(y, y_bin).ravel()
        specificity = tn / (tn + fp)
        sensitivity = tp / (tp + fn)
        optimal_f1 = np.nanmax(f1_all)




        return 'f1',-optimal_f1

    seeds = np.random.randint(5000, 10000, size=10).tolist()
    auc_lst = []
    auc_lst1 = []
    n_estimators_lst =[]
    stratified=True
    debug = True
    param = list_param('xgb_rank')
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
            'seed': seeds[seed_id],
            'objective': 'rank:pairwise',
            'silent': False,
        }
        param.update(params_iter)

        res = xgb.cv(param, xg_train, num_boost_round=5000, folds=gfold_Id, feval=xg_f1,
                     metrics={'auc'}, stratified=True, maximize=False, verbose_eval=50,
                     callbacks=[xgb.callback.print_evaluation(show_stdv=True),
                                xgb.callback.early_stop(50)])

        n_estimators = res.shape[0]

        for n_fold, (train_idx, valid_idx) in enumerate(folds.split(X_train, labels)):
            xgg_train = X_train.iloc[train_idx]
            xgg_valid = X_train.iloc[valid_idx]
            ygg_train = labels[train_idx]
            ygg_valid = labels[valid_idx]

            xgg_train = xgb.DMatrix(xgg_train, label=ygg_train, missing=-1.0)
            xgg_valid = xgb.DMatrix(xgg_valid, missing=-1.0)
            #xg_test = xgb.DMatrix(X_test, missing=-1.0)



            clf = xgb.train(param, xgg_train, num_boost_round=n_estimators, verbose_eval=1)

            oof_preds[valid_idx] = clf.predict(xgg_valid)
            pred = clf.predict(xg_test)
            sub_preds += pred / folds.n_splits

            fpr, tpr, thresholds = metrics.roc_curve(ygg_valid, oof_preds[valid_idx])
            optimal_idx = np.argmax(tpr - fpr)
            optimal_thresholds = thresholds[optimal_idx]

            list_thresholds_global.append(optimal_thresholds)

            sub_preds_local_vote[:, n_fold] = [1 if y_cont > optimal_thresholds else 0 for y_cont in pred]
            oof_preds_local_vote[valid_idx] = [1 if y_cont > optimal_thresholds else 0 for y_cont in oof_preds[valid_idx]]

            fold_raw_importance = pd.DataFrame(list(clf.get_score(importance_type='gain').items()), columns=['feature','importance']).sort_values('importance', ascending=False)
            fold_importance_df = pd.DataFrame()
            fold_importance_df["feature"] = features
            fold_importance_df = pd.merge(fold_importance_df, fold_raw_importance, on='feature', how='left')
            fold_importance_df = fold_importance_df.fillna(value=0)
            fold_importance_df = fold_importance_df.sort_values('importance', ascending=False)
            fold_importance_df["fold"] = n_fold + 1
            fold_importance_df["seed"] = 'seed_' + str(seeds[seed_id])
            feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
            print('Fold %2d AUC : %.6f' % (n_fold + 1, roc_auc_score(ygg_valid, oof_preds[valid_idx])))

            del clf, xgg_train, xgg_valid, ygg_train, ygg_valid
            gc.collect()

        oof_preds_folds[:, seed_id]=oof_preds
        sub_preds_folds[:, seed_id] = sub_preds
        from scipy import stats

        a, b = stats.mode(sub_preds_local_vote, axis=1)
        oof_preds_folds_vote[:, seed_id] = oof_preds_local_vote
        sub_preds_folds_vote[:, seed_id] = a.reshape(-1)
        feature_importance_df_folds = pd.concat([feature_importance_df_folds, feature_importance_df], axis=0)
        auc_lst.append(ml_metrics.auc(y_train, oof_preds))
        auc_lst1.append(roc_auc_score(y_train, oof_preds))
        print('Full AUC score %.6f' % roc_auc_score(y_train, oof_preds))
        print("auc_lst1")
        print(auc_lst1)


    print(list_thresholds_global)
    #oof_preds_folds = pd.DataFrame(oof_preds_folds,columns=['xgb_rank_seed_' + str(seeds[l]) for l in range(len(seeds))])
    #sub_preds_folds = pd.DataFrame(sub_preds_folds,columns=['xgb_rank_seed_' + str(seeds[l]) for l in range(len(seeds))])
    oof_preds_folds_vote = pd.DataFrame(oof_preds_folds_vote,columns=['xgb_rank_seed_' + str(seeds[l]) for l in range(len(seeds))])
    sub_preds_folds_vote = pd.DataFrame(sub_preds_folds_vote,columns=['xgb_rank_seed_' + str(seeds[l]) for l in range(len(seeds))])


    #oof_preds_folds.to_csv("./output/" + task + "_train_stack/xgb_rank.csv", index=False)
    #sub_preds_folds.to_csv("./output/" + task + "_test_stack/xgb_rank.csv", index=False)
    oof_preds_folds_vote.to_csv("./output/" + task + "_train_stack_vote/xgb_rank.csv", index=False)
    sub_preds_folds_vote.to_csv("./output/" + task + "_test_stack_vote/xgb_rank.csv", index=False)
    feature_importance_df_folds=feature_importance_df_folds.sort_values('importance', ascending=False)
    feature_importance_df_folds.to_csv("./output/" + task + "_feature/xgb_rank.csv", index=False)
