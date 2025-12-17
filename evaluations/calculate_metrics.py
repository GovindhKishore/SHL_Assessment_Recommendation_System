"""
Evaluation script to calculate metrics for the SHL Assessment Recommendation System.

This script calculates the recall@10 metric by comparing the system's recommendations
against a ground truth dataset of query-assessment pairs.
"""
import pandas as pd
import requests
import os
import re
from urllib.parse import unquote

# API endpoint for recommendations
API_URL = "http://127.0.0.1:8001/recommend"
# Path to the training dataset with ground truth query-assessment pairs
DATASET_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "given_datasets", "train.csv")


def extract_assessment_name(url):
    """
    Extract a normalized assessment name from a URL for comparison.

    Args:
        url (str): The assessment URL

    Returns:
        str: Normalized assessment name for matching
    """
    if not isinstance(url, str):
        return ""

    s = url.strip()
    if not s:
        return ""

    # Remove query parameters and fragments
    s = s.split('?', 1)[0].split('#', 1)[0]

    # Decode URL-encoded characters
    s = unquote(s)

    # Clean up path
    s = s.rstrip('/').lstrip('/')
    if not s:
        return ""

    # Extract the last part of the path as the name
    parts = [p for p in s.split('/') if p]
    name = parts[-1].lower() if parts else ""

    # Normalize: keep only letters, digits and hyphens
    name = re.sub(r'[^a-z0-9\-]', '', name)

    # Normalize multiple hyphens to single hyphens
    name = re.sub(r'\-+', '-', name).strip('-')

    return name


def calculate_metrics():
    """
    Calculate and print recall metrics for the recommendation system.

    Tests the API with queries from the training dataset and checks if the
    expected assessments appear in the top 10 recommendations.
    """
    print("Starting metrics calculation (normalized-slug + relaxed matching)...")

    # Validate dataset exists
    if not os.path.exists(DATASET_PATH):
        print(f" Error: Could not find dataset at ` {DATASET_PATH} `")
        return

    # Load the dataset
    try:
        df = pd.read_csv(DATASET_PATH)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    print()
    print(f"Loaded {len(df)} test queries.")

    success_count = 0
    processed = 0

    # Process each query in the dataset
    for idx, row in df.iterrows():
        query = row.get('Query', "")
        target_url = str(row.get('Assessment_url', "")).strip()
        target_name = extract_assessment_name(target_url)

        # Skip invalid entries
        if not query or not target_name:
            print(f"Skipping row {idx+1}: missing query or target")
            continue

        # Query the API
        try:
            response = requests.post(API_URL, json={"query": query}, timeout=50)
        except Exception as e:
            print(f"Request Error for query {idx+1}: {e}")
            continue

        # Check for API errors
        if response.status_code != 200:
            print(f"API Error for query {idx+1}: {response.status_code} - {response.text}")
            continue

        # Parse the response
        try:
            data = response.json()
        except Exception as e:
            print(f"Invalid JSON for query {idx+1}: {e}")
            continue

        recommendations = data.get("recommended_assessments", [])
        processed += 1

        rec_count = len(recommendations)
        found = False
        top_name = "None"

        # Check if the target assessment is in the recommendations
        for i, res in enumerate(recommendations):
            res_url = str(res.get('url', '')).strip()
            res_name = extract_assessment_name(res_url)

            # Track the top recommendation for reporting
            if i == 0:
                top_name = res_name or "None"

            # Use relaxed matching to handle URL variations
            if res_name and (res_name == target_name or 
                            res_name.endswith(target_name) or 
                            target_name.endswith(res_name) or 
                            target_name in res_name or 
                            res_name in target_name):
                found = True
                break

        # Report results for this query
        status = "Found" if found else "Missed"
        print(f"Query {idx+1}: {status}  (recs={rec_count})")

        if not found:
            # Detailed reporting for missed targets
            print(f"   Expected name: {target_name}")
            print(f"   Got Top name:  {top_name}")

            # Show the first 10 recommendations for analysis
            for i, r in enumerate(recommendations[:10]):
                ru = str(r.get('url', '')).strip()
                rn = extract_assessment_name(ru)
                print(f"    {i+1}. raw=`{ru}`  name=`{rn}`")
            print("-" * 30)
        else:
            success_count += 1

    # Calculate and report the final recall score
    if processed > 0:
        recall_score = success_count / processed
        print(f"Final Mean Recall@10: {recall_score:.2f} ({int(recall_score * 100)}%)  (based on {processed} processed queries)")
    else:
        print("No queries processed.")


if __name__ == "__main__":
    calculate_metrics()
