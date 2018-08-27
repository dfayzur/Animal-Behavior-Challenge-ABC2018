from stacking_model_cat import stacking_model_cat
from stacking_model_lgb_gbt import stacking_model_lgb_gbt
from stacking_model_lgb_rf import stacking_model_lgb_rf
from stacking_model_xgb_logistic import stacking_model_xgb_logistic
from stacking_model_xgb_rank import stacking_model_xgb_rank
from stacking_model_sk_gbt import stacking_model_sk_gbt
from stacking_model_sk_et import stacking_model_sk_et
from stacking_model_sk_rf import stacking_model_sk_rf
from stacking_model_sk_svc import stacking_model_sk_svc
from stacking_model_sk_gpc import stacking_model_sk_gpc

def models():
    # training with split dataset
    print("Start training all models with split dataset")
    stacking_model_xgb_logistic("split")
    stacking_model_xgb_rank("split")
    stacking_model_cat("split")
    stacking_model_lgb_gbt("split")
    stacking_model_lgb_rf("split")
    stacking_model_sk_gbt("split")
    stacking_model_sk_rf("split")
    stacking_model_sk_et("split")
    stacking_model_sk_svc("split")
    stacking_model_sk_gpc("split")

    print("End training all models with split dataset")

    # training with split dataset
    print("Start training all models with together dataset")
    stacking_model_xgb_logistic("together")
    stacking_model_xgb_rank("together")
    stacking_model_cat("together")
    stacking_model_lgb_gbt("together")
    stacking_model_lgb_rf("together")
    stacking_model_sk_gbt("together")
    stacking_model_sk_rf("together")
    stacking_model_sk_et("together")
    stacking_model_sk_svc("together")
    stacking_model_sk_gpc("together")

    print("End training all models with together dataset")


if __name__ == "__main__":
    # training with split and together dataset
    models()