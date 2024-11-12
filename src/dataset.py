import os
import json
import boto3
import pandas as pd
from botocore import UNSIGNED
from botocore.client import Config

def download_and_convert_ndjson_to_csv(bucket_name, local_folder):
    # Create an S3 client with anonymous access
    s3_client = boto3.client('s3', region_name='us-east-1', config=Config(signature_version=UNSIGNED))

    # List objects in the S3 bucket
    response = s3_client.list_objects_v2(Bucket=bucket_name)

    if 'Contents' not in response:
        print("No files found in the bucket.")
        return

    # Process each JSON file directly from S3 and convert to CSV
    for obj in response['Contents']:
        file_key = obj['Key']
        
        # Check if the file is a JSON file
        if not file_key.endswith('.json'):
            continue

        print(f"Processing {file_key}...")

        # Download the NDJSON content directly from S3
        json_content = s3_client.get_object(Bucket=bucket_name, Key=file_key)['Body'].read().decode('utf-8')
        
        # Convert NDJSON content to CSV
        convert_ndjson_content_to_csv(json_content, file_key, local_folder)

def convert_ndjson_content_to_csv(json_content, file_key, local_folder):
    """Convert NDJSON content to CSV format and save it."""
    try:
        # Split content by lines (each line is a JSON object)
        lines = json_content.splitlines()

        # Convert each JSON line to a dictionary and store in a list
        data = [json.loads(line) for line in lines if line.strip()]

        # Convert the list of dictionaries to a DataFrame
        df = pd.DataFrame(data)

        # Create the local CSV file path
        csv_file_path = os.path.join(local_folder, file_key.replace('.json', '.csv'))
        os.makedirs(os.path.dirname(csv_file_path), exist_ok=True)

        # Save the DataFrame to a CSV file
        df.to_csv(csv_file_path, index=False, encoding='utf-8')
        print(f"Converted to CSV: {csv_file_path}")

    except json.JSONDecodeError as e:
        print(f"JSON decoding error in {file_key}: {e}")
    except Exception as e:
        print(f"Error converting {file_key} to CSV: {e}")

if __name__ == "__main__":
    bucket_name = 'helpful-sentences-from-reviews'
    local_folder = './customer_review_dataset'

    if not os.path.exists(local_folder):
        os.makedirs(local_folder)

    download_and_convert_ndjson_to_csv(bucket_name, local_folder)
    print("All files converted to CSV and saved successfully!")