import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import os
import torch
from time import time
from datasets import load_dataset
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    AutoTokenizer,
    TrainingArguments,
)
from trl import SFTTrainer,setup_chat_format
import json
import pickle
from datasets import Dataset
from prompts.prompts import PromptCollection

model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

compute_dtype = torch.bfloat16
bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True)

time_start = time()

model_config = AutoConfig.from_pretrained(
    model_id,
    trust_remote_code=True,
    max_new_tokens=1024
)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    config=model_config,
    quantization_config=bnb_config,
    device_map='auto',
)
tokenizer = AutoTokenizer.from_pretrained(model_id)
time_end = time()
print(f"Prepare model, tokenizer: {round(time_end-time_start, 3)} sec.")

model, tokenizer = setup_chat_format(model, tokenizer)
model = prepare_model_for_kbit_training(model)


## prepare dataset

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

def convert_to_chat_format(data_point):
    json_data = json.loads(data_point['json_data'])
    original_text = data_point['original_text']

    chat = [
        {"role": "user", "content": original_text},
        {"role": "assistant", "content": json.dumps(json_data, indent=2)}
    ]

    return {"messages": chat}

processed_dataset_messages = dataset.map(
    convert_to_chat_format,
    num_proc= os.cpu_count(),
)

def format_chat_template(row):
    chat = tokenizer.apply_chat_template(row["messages"], tokenize=False)
    return {"text":chat}

processed_dataset = processed_dataset_messages.map(
    format_chat_template,
    num_proc= os.cpu_count(),
)

dataset = processed_dataset.train_test_split(test_size=0.1)

def get_max_length(model):
    conf = model.config
    max_length = None
    for length_setting in ["n_positions", "max_position_embeddings", "seq_length"]:
        max_length = getattr(model.config, length_setting, None)
        if max_length:
            print(f"Found max lenth: {max_length}")
            break
    if not max_length:
        max_length = 1024
        print(f"Using default max length: {max_length}")
    return max_length

## Training

## check this too: https://blog.ovhcloud.com/fine-tuning-llama-2-models-using-a-single-gpu-qlora-and-ai-notebooks/

max_length = get_max_length(model)

peft_config = LoraConfig(
        lora_alpha=64,
        lora_dropout=0.05,
        r=4,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules= ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",]
)


training_arguments = TrainingArguments(
        output_dir="./results_llama3_sft/",
        evaluation_strategy="steps",
        do_eval=True,
        optim="paged_adamw_8bit",
        per_device_train_batch_size=8,
        gradient_accumulation_steps=2,
        per_device_eval_batch_size=8,
        log_level="debug",
        save_steps=1,
        logging_steps=1,
        learning_rate=8e-6,
        eval_steps=1,
        max_steps=20,
        num_train_epochs=20,
        warmup_steps=3,
        lr_scheduler_type="linear",
)


trainer = SFTTrainer(
        model=model,
        train_dataset=dataset['train'],
        eval_dataset=dataset['test'],
        peft_config=peft_config,
        dataset_text_field="text",
        max_seq_length=max_length,
        tokenizer=tokenizer,
        args=training_arguments,
)

trainer.train()
