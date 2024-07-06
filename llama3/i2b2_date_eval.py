from vllm import LLM, SamplingParams
from prompts import PromptCollection
from outlines import models, generate
from pydantic import BaseModel
from outlines_schemas import ExpectedJSONOutputFormat_Dates
import pickle, tqdm, os
import outlines


llm = LLM(model="meta-llama/Meta-Llama-3-8B-Instruct", dtype='float16')
model = models.VLLM(llm)

tokenizer = llm.get_tokenizer()
sampling_params = SamplingParams(temperature=0.1, top_p=0.35, max_tokens=500, stop_token_ids=[tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")])

prompts = PromptCollection()

generator_dates = generate.json(model, ExpectedJSONOutputFormat_Dates, whitespace_pattern=r"[\n\t ]*")

with open('../datasets/i2b2/train_jsons/all_records_train_text.pkl', 'rb') as f:
    loaded_records_text = pickle.load(f)

generated_dates = {}
output_dir = "results_dates"

for id, record in tqdm(loaded_records_text.items(), desc = "Running date Prompt"):
    prompt = prompts.date_prompt(record)
    conversations = tokenizer.apply_chat_template(
        [{'role': 'user', 'content': prompt}],
        tokenize=False,
    )
    generated_date = generator_dates(conversations, max_tokens=500, sampling_params=sampling_params)
    generated_dates[id] = generated_date
    if not os.path.exists(output_dir):
        # Create the directory
        os.makedirs(output_dir)
        print(f"Directory '{output_dir}' created.")
    else:
        print(f"Directory '{output_dir}' already exists.")

    with open(f'{output_dir}/train_date_results.pkl', 'wb') as f:
        pickle.dump(generated_dates, f)
