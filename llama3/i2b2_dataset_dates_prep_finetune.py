import pandas as pd
import json
import pickle
from datasets import Dataset
from prompts.prompts import PromptCollection

with open("../datasets/i2b2/train_jsons/all_records_train_text.pkl", 'rb') as f:
    ground_truth_records_text = pickle.load(f)

with open("../datasets/i2b2/train_jsons/all_records_train.pkl", 'rb') as f:
    ground_truth_records = pickle.load(f)

prompts_obj = PromptCollection()


filtered_dict = {key: [item for item in value if item['TYPE'] == 'DATE'] for key, value in ground_truth_records.items()}


def process_key(data):
    text_only = [item['TEXT'] for item in data]
    json_structure = {"dates": list(set(text_only))}
    json_string = json.dumps(json_structure)
    return json_string

json_list = []
original_text_list = []

# Process each key in filtered_dict
for key, value in filtered_dict.items():
    json_string = process_key(value)
    json_list.append(json_string)
    
    # Add the original text for this key
    original_text = ground_truth_records_text.get(key, "")  # Default to empty string if key not found
    prompt = prompts_obj.date_prompt(original_text)
    original_text_list.append(prompt)

# Create the DataFrame with two columns
df = pd.DataFrame({
    'json_data': json_list,
    'original_text': original_text_list
})

# Convert the DataFrame to a Hugging Face Dataset
dataset = Dataset.from_pandas(df)

print(dataset)
print(dataset[0])

