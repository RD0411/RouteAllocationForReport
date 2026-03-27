import requests
from ml.config import SUPABASE_URL, SUPABASE_KEY

HEADERS = {
    "apikey": SUPABASE_KEY,
    "Authorization": f"Bearer {SUPABASE_KEY}"
}

def fetch_table(table_name, filters=""):
    url = f"{SUPABASE_URL}/rest/v1/{table_name}"

    if filters:
        url += f"?{filters}"

    response = requests.get(url, headers=HEADERS)

    if response.status_code != 200:
        raise Exception(f"Error {response.status_code}: {response.text}")

    return response.json()