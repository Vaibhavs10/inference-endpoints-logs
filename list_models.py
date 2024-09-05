import json
from huggingface_hub.utils._pagination import paginate

endpoint = "https://huggingface.co/api/models"
params = {"inference": "warm"}

results = []
for page in paginate(endpoint, params=params, headers={}):
    results.append(page)

with open('models_data.json', 'w') as f:
    json.dump(results, f, indent=4)