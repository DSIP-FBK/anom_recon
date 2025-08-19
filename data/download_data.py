#!/usr/bin/env python
# coding: utf-8

"""
run this script to download the dataset from Zenodo
"""

import requests

# Zenodo record ID
record_id = "16751720"

# Fetch record metadata
api_url = f"https://zenodo.org/api/records/{record_id}"
response = requests.get(api_url)
response.raise_for_status()
data = response.json()

# Download each file
for file_info in data['files']:
    file_url = file_info['links']['self']
    file_name = file_info['key']

    print(f"Downloading {file_name}...")

    # Stream download
    with requests.get(file_url, stream=True) as r:
        r.raise_for_status()
        with open(file_name, 'wb') as f:
            for chunk in r.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)

print(f"All files downloaded")
