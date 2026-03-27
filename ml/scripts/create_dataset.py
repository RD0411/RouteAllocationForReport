from ml.scripts.fetch_data import load_data
from ml.scripts.feature_engineering import build_features
from ml.config import DATASET_PATH


def main():
    print("🚀 Starting dataset creation...\n")

    # Load data from Supabase
    reports, routes, route_reports, workers, assignments = load_data()

    # Debug logs
    print("📊 DATA SUMMARY:")
    print(f"Reports: {len(reports)}")
    print(f"Routes: {len(routes)}")
    print(f"Route Reports: {len(route_reports)}")
    print(f"Workers: {len(workers)}")
    print(f"Assignments: {len(assignments)}\n")

    # Check minimum data requirements
    if reports.empty:
        print("❌ No reports found. Add data in waste_reports table.")
        return

    if routes.empty:
        print("❌ No routes found. Add data in routes table.")
        return

    if workers.empty:
        print("❌ No workers found. Add data in workers table.")
        return

    # Build dataset
    print("⚙️ Generating features...")
    df = build_features(reports, routes, route_reports, workers, assignments)

    # Handle empty dataset
    if df.empty:
        print("\n❌ Dataset is empty!")
        print("👉 Possible reasons:")
        print("   - No route_reports mapping")
        print("   - No route_assignments")
        print("   - No matching data between tables")
        print("\n💡 Temporary fix: modify feature_engineering to use all reports as spots")
        return

    # Save dataset
    df.to_csv(DATASET_PATH, index=False)

    print("\n✅ Dataset created successfully!")
    print(f"📁 Saved at: {DATASET_PATH}")
    print(f"📦 Total rows: {len(df)}")


if __name__ == "__main__":
    main()