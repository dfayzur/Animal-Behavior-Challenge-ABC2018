
def list_param(algo='xgb_rank'):
    params_all = dict()

    params_all['xgb_rank'] = {'max_depth': int(5.0),
                              'learning_rate': 0.025,
                              'gamma': 3.0,
                              'min_child_weight': 2.0,
                              'reg_alpha': 0.025,
                              'reg_lambda': 0.025,
                              'subsample': 0.25,
                              'colsample_bytree': 0.25
                              } #0.767530


    params_all['xgb_binary'] = {'reg_lambda': 0.6514270867678106,
                               'colsample_bytree': 0.11698581408775477,
                               'max_depth': int(4.001945385072979),
                               'learning_rate': 0.028727453282415254,
                               'min_child_weight': 1.549711065449091,
                               'reg_alpha': 0.7783910130481886,
                               'subsample': 0.967731305704434,
                               'gamma': 0.047179798063114786
                               } # 0.767530

    params_all['lgb_gbdt'] = {'num_leaves': int(19.14803706502223),
                              'min_data_in_leaf': int(19.117153693674837),
                              'learning_rate': 0.028537915641618315,
                              'min_sum_hessian_in_leaf': 0.009783416296025901,
                              'bagging_fraction': 0.8715539756922579,
                              'feature_fraction': 0.4495812455596912,
                              'lambda_l1': 3.1250699911571402,
                              'lambda_l2': 3.864868357755003,
                              'min_gain_to_split': 0.1275297299314797,
                              'max_depth': int(14.522791339641064)
                              }  # 0.767029


    params_all['lgb_rf'] = {'min_data_in_leaf': int(19.954688804498467),
                            'learning_rate': 0.005899265439506244,
                            'bagging_fraction': 0.8107906815467697,
                            'max_depth': int(8.492711215183968),
                            'min_gain_to_split': 0.3366842032801287,
                            'lambda_l2': 3.1393381597530117,
                            'num_leaves': int(16.47082171582435),
                            'lambda_l1': 2.6080697805932496,
                            'min_sum_hessian_in_leaf': 0.0036560302506285177,
                            'feature_fraction': 0.18854525999441535
                            }  # 0.755920


    params_all['cat'] = {'colsample_bylevel': 0.2710530847696454,
                         'subsample': 0.6253678727765083,
                         'learning_rate': 0.015810572483020525,
                         'depth': int(8.200872481618028),
                         'l2_leaf_reg': int(4.408465731987901)
                         }  # 0.734398


    params_all['sk_gbt'] = {'n_estimators': int(183.6444104914655),
                            'min_samples_split': int(11.076530369743017),
                            'min_samples_leaf': int(1.0127838535995122),
                            'max_depth': int(8.906856852166516),
                            'max_features': 0.17116115145143493,
                            'subsample': 0.9908104285300018
                            }  # 0.765193

    params_all['sk_rf'] = {'min_samples_split': 12.350407247772479,
                           'min_samples_leaf': 9.05611775956333,
                           'n_estimators': 141.2649437724456,
                           'max_depth': 6.889695915709787,
                           'max_features': 0.22697537083729424
                           }  # 0.759317



    params_all['sk_et'] = {'min_samples_leaf': 5.472615337294361,
                           'max_depth': 7.011463437331648,
                           'max_features': 0.402087928396633,
                           'min_samples_split': 7.2135271715193685,
                           'n_estimators': 102.65949812569075
                           }   # 0.752338

    params_all['sk_svc'] = {'gamma': 0.1208412754505453,
                            'C': 38.32334595925415,
                            'max_iter': 3684.4637965095753,
                            'tol': 1.0205731122700548
                            }  # 0.725369

    params_all['sk_gpc'] = {'max_iter_predict': int(100.0),
                            'n_restarts_optimizer': int(0.0)
                            }  # 0.714567


    return params_all[algo]
