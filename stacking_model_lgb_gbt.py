import numpy as np
import pandas as pd
import gc
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import KFold, StratifiedKFold
import warnings
from sklearn.metrics import confusion_matrix
from sklearn import metrics


import ml_metrics
np.random.RandomState(2018)
np.random.seed(2018)
import random
random.seed(2018)


warnings.simplefilter(action='ignore', category=FutureWarning)

np.random.randint(5000, 10000, size=10).tolist()
random.sample(range(5000, 10000), 10)

def stacking_model_lgb_gbt(task='together'):
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

    gc.collect()

    #lightgbm
    X_train = train_df[predictors].values

    labels = train_df['label']


    def xg_f1(preds, train_data):
        yhat=preds

        dtrain = train_data

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




        return 'f1',-optimal_f1, False

    xg_train = lgb.Dataset(train_df[predictors].values,
                               label=train_df[target].values,
                               feature_name=predictors
                               )

    seeds = np.random.randint(5000, 10000, size=10).tolist()
    auc_lst = []
    auc_lst1 = []
    n_estimators_lst =[]
    stratified=True
    debug = True
    param = list_param('lgb_gbdt')
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
            'max_bin': 63,  # fixed #int
            'save_binary': True,  # fixed
            'seed': seeds[seed_id],
            'feature_fraction_seed': seeds[seed_id],
            'bagging_seed': seeds[seed_id],
            'drop_seed': seeds[seed_id],
            'data_random_seed': seeds[seed_id],
            'objective': 'binary',
            'boosting_type': 'gbdt',
            'verbose': 1,
            'metric': 'auc',

        }
        param.update(params_iter)

        bst1 = lgb.cv(param,
                      xg_train,
                      num_boost_round=5000,
                      early_stopping_rounds=50, folds=gfold_Id)

        res0 = pd.DataFrame(bst1)

        n_estimators = res0.shape[0]

        for n_fold, (train_idx, valid_idx) in enumerate(folds.split(X_train, labels)):
            xgg_train = lgb.Dataset(data=train_df[predictors].iloc[train_idx],
                                 label=train_df[target].iloc[train_idx],
                                 free_raw_data=False, silent=True)
            xgg_valid = lgb.Dataset(data=train_df[predictors].iloc[valid_idx],
                                 label=train_df[target].iloc[valid_idx],
                                 free_raw_data=False, silent=True)

            clf = lgb.train(param,
                            xgg_train,
                              num_boost_round=n_estimators,
                              # fobj=loglikelood,
                              # feval=binary_error,
                              verbose_eval=1,
                              )

            oof_preds[valid_idx] = clf.predict(xgg_valid.data)
            pred = clf.predict(test_df[predictors])
            sub_preds += pred / folds.n_splits

            fpr, tpr, thresholds = metrics.roc_curve(xgg_valid.label, oof_preds[valid_idx])
            optimal_idx = np.argmax(tpr - fpr)
            optimal_thresholds = thresholds[optimal_idx]

            list_thresholds_global.append(optimal_thresholds)

            sub_preds_local_vote[:, n_fold] = [1 if y_cont > optimal_thresholds else 0 for y_cont in pred]
            oof_preds_local_vote[valid_idx] = [1 if y_cont > optimal_thresholds else 0 for y_cont in oof_preds[valid_idx]]

            fold_importance_df = pd.DataFrame()
            fold_importance_df["feature"] = clf.feature_name()
            fold_importance_df["importance"] = clf.feature_importance(importance_type='gain')
            fold_importance_df = fold_importance_df.fillna(value=0)
            fold_importance_df = fold_importance_df.sort_values('importance', ascending=False)
            fold_importance_df["fold"] = n_fold + 1
            fold_importance_df["seed"] = 'seed_' + str(seeds[seed_id])
            feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
            print('Fold %2d AUC : %.6f' % (n_fold + 1, roc_auc_score(xgg_valid.label, oof_preds[valid_idx])))

            del clf, xgg_train, xgg_valid
            gc.collect()

        oof_preds_folds[:, seed_id]=oof_preds
        sub_preds_folds[:, seed_id] = sub_preds
        from scipy import stats

        a, b = stats.mode(sub_preds_local_vote, axis=1)
        oof_preds_folds_vote[:, seed_id] = oof_preds_local_vote
        sub_preds_folds_vote[:, seed_id] = a.reshape(-1)
        feature_importance_df_folds = pd.concat([feature_importance_df_folds, feature_importance_df], axis=0)
        auc_lst.append(ml_metrics.auc(xg_train.label, oof_preds))
        auc_lst1.append(roc_auc_score(xg_train.label, oof_preds))
        print('Full AUC score %.6f' % roc_auc_score(xg_train.label, oof_preds))
        print("auc_lst1")
        print(auc_lst1)


    print(list_thresholds_global)
    #oof_preds_folds = pd.DataFrame(oof_preds_folds,columns=['lgb_gbt_seed_' + str(seeds[l]) for l in range(len(seeds))])
    #sub_preds_folds = pd.DataFrame(sub_preds_folds,columns=['lgb_gbt_seed_' + str(seeds[l]) for l in range(len(seeds))])
    oof_preds_folds_vote = pd.DataFrame(oof_preds_folds_vote,columns=['lgb_gbt_seed_' + str(seeds[l]) for l in range(len(seeds))])
    sub_preds_folds_vote = pd.DataFrame(sub_preds_folds_vote,columns=['lgb_gbt_seed_' + str(seeds[l]) for l in range(len(seeds))])

    #oof_preds_folds.to_csv("./output/" + task + "_train_stack/lgb_gbt.csv", index=False)
    #sub_preds_folds.to_csv("./output/" + task + "_test_stack/lgb_gbt.csv", index=False)
    oof_preds_folds_vote.to_csv("./output/" + task + "_train_stack_vote/lgb_gbt.csv", index=False)
    sub_preds_folds_vote.to_csv("./output/" + task + "_test_stack_vote/lgb_gbt.csv", index=False)
    feature_importance_df_folds=feature_importance_df_folds.sort_values('importance', ascending=False)
    feature_importance_df_folds.to_csv("./output/" + task + "_feature/lgb_gbt.csv", index=False)
