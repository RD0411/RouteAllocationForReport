import pandas as pd
import numpy as np
from ml.utils.distance import haversine
from ml.config import DATASET_PATH


# Pune bounding box (approx)
LAT_MIN, LAT_MAX = 18.40, 18.70
LNG_MIN, LNG_MAX = 73.70, 74.00


def random_location():
    return (
        np.random.uniform(LAT_MIN, LAT_MAX),
        np.random.uniform(LNG_MIN, LNG_MAX)
    )


def generate_dataset(num_reports=200, num_routes=10):
    dataset = []

    # Generate routes (centroids)
    routes = []
    for i in range(num_routes):
        lat, lng = random_location()
        routes.append({
            "route_id": f"route_{i}",
            "centroid_lat": lat,
            "centroid_lng": lng
        })

    # Generate workers (1 per route)
    workers = []
    for i in range(num_routes):
        lat, lng = random_location()
        workers.append({
            "worker_id": f"worker_{i}",
            "lat": lat,
            "lng": lng,
            "status": np.random.choice(["available", "busy", "offline"])
        })

    # Generate reports
    for _ in range(num_reports):
        report_lat, report_lng = random_location()

        route_candidates = []

        for i in range(num_routes):
            route = routes[i]
            worker = workers[i]

            d_route = haversine(report_lat, report_lng,
                                route["centroid_lat"], route["centroid_lng"])

            d_worker = haversine(report_lat, report_lng,
                                 worker["lat"], worker["lng"])

            route_candidates.append({
                "report_lat": report_lat,
                "report_lng": report_lng,
                "route_id": route["route_id"],
                "route_centroid_lat": route["centroid_lat"],
                "route_centroid_lng": route["centroid_lng"],
                "driver_lat": worker["lat"],
                "driver_lng": worker["lng"],
                "distance_report_to_route": d_route,
                "distance_report_to_driver": d_worker,
                "route_load": np.random.randint(1, 10),
                "driver_status": worker["status"],
                "label": 0
            })

        # Assign best route (minimum distance)
        best_idx = np.argmin(
            [r["distance_report_to_route"] for r in route_candidates]
        )
        route_candidates[best_idx]["label"] = 1

        dataset.extend(route_candidates)

    return pd.DataFrame(dataset)


def main():
    print("🚀 Generating synthetic dataset...")

    df = generate_dataset(num_reports=300, num_routes=15)

    df.to_csv(DATASET_PATH, index=False)

    print("✅ Synthetic dataset created!")
    print(f"📦 Rows: {len(df)}")
    print(f"📁 Saved at: {DATASET_PATH}")


if __name__ == "__main__":
    main()