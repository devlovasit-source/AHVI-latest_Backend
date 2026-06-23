import os
import requests
import boto3
from botocore.config import Config
from dotenv import load_dotenv
import mimetypes
import uuid

def main():
    load_dotenv()
    
    r2_url = os.getenv("R2_S3_API_URL")
    access_key = os.getenv("R2_ACCESS_KEY_ID")
    secret_key = os.getenv("R2_SECRET_ACCESS_KEY")
    bucket = os.getenv("R2_BUCKET_WARDROBE", "wardrobe-images")
    
    if not all([r2_url, access_key, secret_key, bucket]):
        print("Missing required R2 environment variables.")
        return
        
    print(f"Connecting to R2 URL: {r2_url}")
    print(f"Target Bucket: {bucket}")
    
    # Initialize S3 client for R2
    s3_client = boto3.client(
        's3',
        endpoint_url=r2_url,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name="auto",
        config=Config(signature_version="s3v4")
    )
    
    # Files to upload
    files_to_upload = [
        r"C:\tmp\ahvi_rmbg_trousers_checker.png",
        r"C:\tmp\ahvi_rmbg_trousers_direct.png",
        r"C:\tmp\ahvi_route_rmbg_result.png",
        r"C:\tmp\ahvi_backend_rmbg_result.png",
        r"C:\tmp\ahvi_rmbg_result.png",
        r"C:\tmp\ahvi_rmbg_test.png"
    ]
    
    for filepath in files_to_upload:
        if not os.path.exists(filepath):
            print(f"File not found: {filepath}")
            continue
            
        filename = os.path.basename(filepath)
        content_type, _ = mimetypes.guess_type(filepath)
        if not content_type:
            content_type = "image/png"
            
        # Using a fixed prefix so they're easy to identify
        s3_key = f"cleaned_assets/{filename}"
        
        print(f"Uploading {filename} to {s3_key}...")
        try:
            with open(filepath, 'rb') as f:
                s3_client.put_object(
                    Bucket=bucket,
                    Key=s3_key,
                    Body=f,
                    ContentType=content_type
                )
            print(f"Success! Uploaded {filename}")
        except Exception as e:
            print(f"Error uploading {filename}: {e}")

if __name__ == "__main__":
    main()
