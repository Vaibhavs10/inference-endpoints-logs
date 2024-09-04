import requests
import json

def persist_models_to_json(url):
    response = requests.get(url)
    if response.status_code == 200:
        models_data = response.json()
        with open('models_data.json', 'w') as json_file:
            json.dump(models_data, json_file)
        print("Models data persisted to models_data.json successfully.")
    else:
        print(f"Failed to fetch models data. Status code: {response.status_code}")

url = "https://huggingface.co/api/models?inference=warm"

persist_models_to_json(url)
