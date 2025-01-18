import os
import json
from tqdm import tqdm
from openai import OpenAI

from src.distractors.distractors_pipeline import prompt_distractors, parse_chatgpt_response
# Define the OpenAI API function
client = OpenAI()
model_params = {
    "model": "gpt-4",
    "temperature": 0.7,
}

# Load the JSON file
json_path = "../data/opentom.json"
with open(json_path, "r") as file:
    data = json.load(file)  # Load as a list of dictionaries


# Process each entry in the JSON data
for entry in tqdm(data[:5], desc="Processing entries", unit="entry"):
    context = entry.get("narrative", "")
    original_question = entry.get("question", "").get("question", "")
    original_answer_type = entry.get("question", "").get("type", "")
    original_answer = entry.get("question", "").get("answer", "")

    # Get ChatGPT response
    response = prompt_distractors(context, original_question, original_answer_type, original_answer, client, model_params=model_params)

    # Parse the response to extract the new question and answer
    entry['question']['new_question'], entry['question']['new_answer'] = parse_chatgpt_response(response)

# Save the updated JSON file
updated_json_path = "../data/updated_dataset.json"
with open(updated_json_path, "w") as file:
    json.dump(data, file, indent=4)

print(f"Updated JSON saved to {updated_json_path}")