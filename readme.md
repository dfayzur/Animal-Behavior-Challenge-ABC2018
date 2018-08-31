Code of the winning entry to the [Animal Behavior Challenge (ABC2018) for understanding animal behavior](https://competitions.codalab.org/competitions/16283). Our approach is described in [our paper](https://arxiv.org/abs/1808.08613).

## Dependencies

The dependencies can be installed with:

pip install -r requirements.txt

## Structure

Here is a brief description of the Python files in the archive:

* `readme.md`: this file.
* `requirements.txt` contains list of packages need to install to run all models.
* `data/` contains original training and testing dataset along with newly created dataset.
* `doc/` contains the technical report of the winning solution.
* `output/` contains intermidiate outputs from trained model to use in the final submission.
* `Submission/` contains the submission file to score.
* `feature_engineering_day_night_split.py`: creates new dataset with feature engineering by splitting original dataset to day and night. This will create a new dataset.
* `feature_engineering_day_night_together.py` : creates new dataset with feature engineering considering the original dataset without day and night split. This will create a new dataset.
* `stacking_model_*.py` : scripts for various modeling on the datasets 
  * `stacking_model_cat.py` : train a CatBoost model for binary classification
  * `stacking_model_lgb_gbt.py` : train a LightGBM model with traditional Gradient Boosting Decision Tree for for binary classification
  * `stacking_model_lgb_rf.py` : train a LightGBM model with Random Forest for for binary classification
  * `stacking_model_xgb_logistic.py` : train a XGBoost model with logistic regression for binary classification
  * `stacking_model_xgb_rank.py` : train a XGBoost model to do ranking task by minimizing the pairwise loss
  * `stacking_model_sk_gbt` : train a GradientBoostingClassifier model from scikit-learn to do binary classification
  * `stacking_model_sk_rf.py` : train a RandomForestClassifier model from scikit-learn to do binary classification
  * `stacking_model_sk_et.py` : train a ExtraTreesClassifier model from scikit-learn to do binary classification
  * `stacking_model_sk_svc.py` : train a Support Vector Machines model from scikit-learn to do binary classification
  * `stacking_model_sk_gpc.py` : train a Gaussian process classification model from scikit-learn to do binary classification
* `training.py` : calls `stacking_model_*.py` scripts for two datset by passing parameter ("split" and "together")
* `split_fn_param.py` contains the optimal parameters from 5 fold cross validation, used in the stacking_model_*.py
* `together_fn_param.py` contains the optimal parameters from 5 fold cross validation, used in the stacking_model_*.py
* `generate_submission.py` generates final submission file from ensemble of files

## How to generate submission file?

Run following scripts in sequence:

1. Run `feature_engineering_day_night_split.py` to generate the csv files `train_df_day_night_split.csv` and `test_df_day_night_split.csv`.
2. Run `feature_engineering_day_night_together.py` to generate the csv files `train_df_day_night_together.csv` and `train_df_day_night_together.csv`.
3. Run `training.py` to train all models on datasets created from previous setps.
4. Run `generate_submission.py` to generate the final submission file. 
