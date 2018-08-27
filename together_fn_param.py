
def list_param(algo='xgb_rank'):
    params_all = dict()

    params_all['xgb_rank'] = {'gamma': 3.0,
                              'reg_alpha': 0.9721340121586061,
                              'colsample_bytree': 0.08528687770163024,
                              'learning_rate': 0.001,
                              'reg_lambda': 1.0,
                              'subsample': 1.0,
                              'min_child_weight': 2.0,
                              'max_depth': int(8.0)
                              } #0.771361


    params_all['xgb_binary'] = {'reg_alpha': 0.038821221372636106,
                                'learning_rate': 0.02276485943054031,
                                'colsample_bytree': 0.21718746841314424,
                                'reg_lambda': 0.9832807166317002,
                                'min_child_weight': 0.9800596038097933,
                                'max_depth': int(7.867647705711956),
                                'subsample': 0.4678236924807951,
                                'gamma': 2.9338973134209825
                                }  # 0.768704

    params_all['lgb_gbdt'] = {'min_data_in_leaf': int(6.705927094563151),
                              'learning_rate': 0.029189690796221572,
                              'min_gain_to_split': 0.8987919640903325,
                              'num_leaves': int(7.300081474074251),
                              'bagging_fraction': 0.9362406521860581,
                              'min_sum_hessian_in_leaf': 0.004225626950212549,
                              'max_depth': int(8.01136132698523),
                              'lambda_l2': 3.277516623539276,
                              'feature_fraction': 0.09513875081107831,
                              'lambda_l1': 0.14419850286645275
                              }  # 0.765479



    params_all['lgb_rf'] = {'lambda_l2': 1.9025926607303634,
                            'max_depth': int(5.103830784180557),
                            'min_sum_hessian_in_leaf': 0.0034034318666464752,
                            'num_leaves': int(18.399921402552735),
                            'min_data_in_leaf': int(13.102578366666636),
                            'bagging_fraction': 0.935828208366019,
                            'feature_fraction': 0.21967301683484086,
                            'min_gain_to_split': 0.9329649747591963,
                            'lambda_l1': 0.08025354960797093,
                            'learning_rate': 0.005831936393793437
                            }  # 0.758018



    params_all['cat'] = {'l2_leaf_reg': int(2.3167928851779505),
                         'subsample': 0.4,
                         'colsample_bylevel': 0.5,
                         'depth': int(3.306839197234326),
                         'learning_rate': 0.03
                         }  # 0.732773



    params_all['sk_gbt'] = {'n_estimators': int(198.20449206532143),
                            'min_samples_split': int(2.444501446273955),
                            'min_samples_leaf': int(1.3214435870063985),
                            'max_depth': int(9.968855292651625),
                            'max_features': 0.4531612486874677,
                            'subsample': 0.782706598682644
                            }  # 0.768177


    params_all['sk_rf'] = {'max_features': 0.4467067262770984,
                           'min_samples_split': 13.520186399843391,
                           'max_depth': 9.721799986383331,
                           'n_estimators': 122.42096237667002,
                           'min_samples_leaf': 9.948266804644334
                           }  # 0.761317

    params_all['sk_et'] = {'max_depth': 8.893135311893008,
                           'min_samples_leaf': 9.506486315819977,
                           'min_samples_split': 2.2416963145840025,
                           'max_features': 0.4543493802866553,
                           'n_estimators': 160.2736184375186
                            }  # 0.761673

    params_all['sk_svc'] = {'gamma': 0.001,
                            'C': 100.0,
                            'max_iter': 2054.2866550174886,
                            'tol': 1e-05
                            }  # 0.725569

    params_all['sk_gpc'] = {'max_iter_predict': int(100.0),
                            'n_restarts_optimizer': int(0.0)
                            }  # 0.717157

    return params_all[algo]
