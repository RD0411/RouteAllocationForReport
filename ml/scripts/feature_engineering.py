import pandas as pd
from ml.utils.distance import haversine

def build_features(reports, routes, route_reports, workers, assignments):

    dataset = []

    for _, report in reports.iterrows():
        for _, route in routes.iterrows():

            rr = route_reports[route_reports['route_id'] == route['id']]
            spot_ids = rr['report_id'].values

            spots = reports.copy()

            if len(spots) == 0:
                continue

            centroid_lat = spots['latitude'].mean()
            centroid_lng = spots['longitude'].mean()

            assign = assignments[assignments['route_id'] == route['id']]
            if assign.empty:
                continue

            worker_id = assign.iloc[0]['worker_id']
            worker = workers[workers['id'] == worker_id].iloc[0]

            d_route = haversine(report['latitude'], report['longitude'],
                                centroid_lat, centroid_lng)

            d_worker = haversine(report['latitude'], report['longitude'],
                                 worker['latitude'], worker['longitude'])

            dataset.append({
                "report_lat": report['latitude'],
                "report_lng": report['longitude'],
                "route_id": route['id'],
                "route_centroid_lat": centroid_lat,
                "route_centroid_lng": centroid_lng,
                "driver_lat": worker['latitude'],
                "driver_lng": worker['longitude'],
                "distance_report_to_route": d_route,
                "distance_report_to_driver": d_worker,
                "route_load": len(spots),
                "driver_status": worker['status'],
                "label": 0
            })

    df = pd.DataFrame(dataset)

    if df.empty:
        print("❌ Dataset is empty. Check your database data.")
        return df

    # Label best route
    for _, group in df.groupby(['report_lat','report_lng']):
        min_idx = group['distance_report_to_route'].idxmin()
        df.loc[min_idx, 'label'] = 1

    return df