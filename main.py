import json
import requests
from huggingface_hub import InferenceClient
import time

# Load models data
with open('models_data.json') as f:
    models_data = json.load(f)

# Initialize InferenceClient
client = InferenceClient()

results = []

for model in models_data:
    model_id = model['model_id']
    task = model['task']
    
    print(f"Processing model: {model_id}")
    
    try:
        if task == 'text-generation':
            response = client.text_generation("Hello, how are you?", model=model_id)
        elif task == 'summarization':
            response = client.summarization("Your long text to summarize here", model=model_id)
        elif task == 'question-answering':
            response = client.question_answering("What is the capital of France?", "France is a country in Europe.", model=model_id)
        elif task == 'image-classification':
            with open('sample_image.jpg', 'rb') as image_file:
                response = client.image_classification(image_file, model=model_id)
        # Add more task types as needed
        
        success = True
        error_message = None
    except Exception as e:
        success = False
        error_message = str(e)
        response = None
    
    result = {
        'model_id': model_id,
        'task': task,
        'success': success,
        'error_message': error_message,
        'response': response
    }
    
    results.append(result)
    
    # Add a small delay to avoid rate limiting
    time.sleep(1)

# Save results to file
with open('inference_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print("Inference completed. Results saved to inference_results.json")
