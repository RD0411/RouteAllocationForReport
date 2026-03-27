from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import lightgbm as lgb
import requests

from ml.utils.db import fetch_table
from ml.utils.distance import haversine
from ml.config import MODEL_PATH, SUPABASE_URL, SUPABASE_KEY

app = FastAPI()

# ================================
# LOAD MODEL
# ================================
model = lgb.Booster(model_file=MODEL_PATH)


# ================================
# REQUEST MODEL
# ================================
class RouteRequest(BaseModel):
    report_id: str


# ================================
# INSERT FUNCTION (AUTO ASSIGN)
# ================================
def insert_route_report(route_id, report_id):
    url = f"{SUPABASE_URL}/rest/v1/route_reports"

    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json"
    }

    data = {
        "route_id": route_id,
        "report_id": report_id
    }

    response = requests.post(url, json=data, headers=headers)

    print("Inserted into DB:", response.status_code, response.text)


# ================================
# MAIN API
# ================================
@app.post("/assign-route")
def assign_route(req: RouteRequest):

    print("\n---- NEW REQUEST ----")
    print("Report ID:", req.report_id)

    # 🔹 Fetch report
    report_data = fetch_table("waste_reports", f"id=eq.{req.report_id}")

    if not report_data:
        raise HTTPException(status_code=404, detail="Report not found")

    report = report_data[0]

    report_lat = report["latitude"]
    report_lng = report["longitude"]

    # 🔹 Fetch required data
    routes = pd.DataFrame(fetch_table("routes", "status=eq.assigned"))
    route_reports = pd.DataFrame(fetch_table("route_reports"))
    workers = pd.DataFrame(fetch_table("workers"))
    assignments = pd.DataFrame(fetch_table("route_assignments"))

    if routes.empty:
        raise HTTPException(status_code=400, detail="No routes available")

    results = []

    for _, route in routes.iterrows():

        rr = route_reports[route_reports['route_id'] == route['id']]
        spot_ids = rr['report_id'].values

        if len(spot_ids) == 0:
            continue

        # Fetch all reports (for centroid calculation)
        all_reports = pd.DataFrame(fetch_table("waste_reports"))
        spots = all_reports[all_reports['id'].isin(spot_ids)]

        if spots.empty:
            continue

        # Route centroid
        centroid_lat = spots['latitude'].mean()
        centroid_lng = spots['longitude'].mean()

        # Assigned worker
        assign = assignments[assignments['route_id'] == route['id']]
        if assign.empty:
            continue

        worker_id = assign.iloc[0]['worker_id']
        worker = workers[workers['id'] == worker_id]

        if worker.empty:
            continue

        worker = worker.iloc[0]

        # Distance calculations
        d_route = haversine(report_lat, report_lng,
                            centroid_lat, centroid_lng)

        d_worker = haversine(report_lat, report_lng,
                             worker['latitude'], worker['longitude'])

        results.append({
            "route_id": route['id'],  # keep for output only
            "report_lat": report_lat,
            "report_lng": report_lng,
            "route_centroid_lat": centroid_lat,
            "route_centroid_lng": centroid_lng,
            "driver_lat": worker['latitude'],
            "driver_lng": worker['longitude'],
            "distance_report_to_route": d_route,
            "distance_report_to_driver": d_worker,
            "route_load": len(spots),
            "driver_status": worker['status']
        })

    df = pd.DataFrame(results)

    if df.empty:
        raise HTTPException(status_code=400, detail="No valid routes found")

    # Encode categorical
    df["driver_status"] = df["driver_status"].map({
        "available": 1,
        "busy": 0,
        "offline": -1
    })

    # 🚨 Remove non-numeric column before prediction
    df_model = df.drop(columns=["route_id"])

    # 🔹 Predict
    scores = model.predict(df_model)
    df["score"] = scores

    # 🔹 Select best route
    best = df.sort_values("score", ascending=False).iloc[0]

    print("Best Route:", best["route_id"])

    # 🔹 Prevent duplicate assignment
    existing = fetch_table(
        "route_reports",
        f"report_id=eq.{req.report_id}"
    )

    if not existing:
        insert_route_report(best["route_id"], req.report_id)
    else:
        print("⚠️ Report already assigned")

    return {
        "best_route_id": best["route_id"],
        "score": float(best["score"])
    }