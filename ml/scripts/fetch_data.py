import pandas as pd
from ml.utils.db import fetch_table

def load_data():
    reports = pd.DataFrame(fetch_table("waste_reports", "status=eq.pending"))
    routes = pd.DataFrame(fetch_table("routes", "status=eq.assigned"))
    route_reports = pd.DataFrame(fetch_table("route_reports"))
    workers = pd.DataFrame(fetch_table("workers", "status=eq.available"))
    assignments = pd.DataFrame(fetch_table("route_assignments"))

    return reports, routes, route_reports, workers, assignments