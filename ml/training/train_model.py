import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from ml.config import DATASET_PATH, MODEL_PATH

df = pd.read_csv(DATASET_PATH)

X = df.drop(columns=["label", "route_id"])
y = df["label"]

X["driver_status"] = X["driver_status"].map({
    "available": 1,
    "busy": 0,
    "offline": -1
})

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

train_data = lgb.Dataset(X_train, label=y_train)

params = {
    "objective": "binary",
    "metric": "binary_logloss",
    "learning_rate": 0.05,
    "num_leaves": 31
}

model = lgb.train(params, train_data, num_boost_round=100)

model.save_model(MODEL_PATH)

print("✅ Model saved:", MODEL_PATH)