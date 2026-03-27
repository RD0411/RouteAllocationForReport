from fastapi import FastAPI
import pandas as pd
import lightgbm as lgb
from ml.config import MODEL_PATH

app = FastAPI()

model = lgb.Booster(model_file=MODEL_PATH)

@app.post("/predict-route")
def predict(data: dict):

    df = pd.DataFrame([data])

    df["driver_status"] = df["driver_status"].map({
        "available": 1,
        "busy": 0,
        "offline": -1
    })

    prediction = model.predict(df)

    return {
        "score": float(prediction[0])
    }