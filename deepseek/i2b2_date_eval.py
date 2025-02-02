from vllm import LLM, SamplingParams # type: ignore
from prompts.prompts import PromptCollection
import pickle, os
from tqdm import tqdm
from postprocess_response import process_response


llm = LLM(model="deepseek-ai/DeepSeek-R1-Distill-Llama-8B")

tokenizer = llm.get_tokenizer()
sampling_params = SamplingParams(temperature=0.1, top_p=0.35, max_tokens=3000)

prompts_obj = PromptCollection()

with open('../datasets/i2b2/train_jsons/all_records_train_text.pkl', 'rb') as f:
    loaded_records_text = pickle.load(f)

generated_dates = {}
output_dir = "results_dates"

if not os.path.exists(output_dir):
    # Create the directory
    os.makedirs(output_dir)
    print(f"Directory '{output_dir}' created.")
else:
    print(f"Directory '{output_dir}' already exists.")

for id, record in tqdm(loaded_records_text.items(), desc = "Running date Prompt"):
    prompt = prompts_obj.date_prompt(record)
    try:
        generated_date = llm.generate([prompt], sampling_params)
        generated_date = process_response(generated_date[0].outputs[0].text)
    except Exception as e:
        print(f"Error occurred for record {id}: {e}")
        generated_dates[id] = {}
        continue
    generated_dates[id] = generated_date

    with open(f'{output_dir}/train_date_results.pkl', 'wb') as f:
        pickle.dump(generated_dates, f)
