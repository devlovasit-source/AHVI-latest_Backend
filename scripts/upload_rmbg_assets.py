import os
import requests
from dotenv import load_dotenv

def main():
    load_dotenv()
    
    endpoint = os.getenv("APPWRITE_ENDPOINT", "https://fra.cloud.appwrite.io/v1")
    project_id = os.getenv("APPWRITE_PROJECT_ID")
    api_key = os.getenv("APPWRITE_API_KEY")
    
    if not all([endpoint, project_id, api_key]):
        print("Missing required environment variables.")
        return
        
    headers = {
        "X-Appwrite-Project": project_id,
        "X-Appwrite-Key": api_key
    }
    
    # Files to upload
    files_to_upload = [
        r"C:\tmp\ahvi_rmbg_trousers_checker.png",
        r"C:\tmp\ahvi_rmbg_trousers_direct.png",
        r"C:\tmp\ahvi_route_rmbg_result.png",
        r"C:\tmp\ahvi_backend_rmbg_result.png",
        r"C:\tmp\ahvi_rmbg_result.png",
        r"C:\tmp\ahvi_rmbg_test.png"
    ]
    
    # Fetch list of buckets
    url = f"{endpoint}/storage/buckets"
    resp = requests.get(url, headers=headers)
    if resp.status_code == 200:
        buckets = resp.json().get('buckets', [])
        print("Available buckets:")
        for b in buckets:
            print(f"- {b['name']} ({b['$id']})")
    else:
        print(f"Failed to fetch buckets: {resp.status_code} - {resp.text}")
        return

if __name__ == "__main__":
    main()
