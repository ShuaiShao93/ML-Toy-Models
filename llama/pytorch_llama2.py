from datasets import load_dataset
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

base_model_id = "meta-llama/Llama-2-7b-chat-hf"
tokenizer = AutoTokenizer.from_pretrained(
    base_model_id,
    padding_side="left", # More efficient for decoder only model
    # add_bos_token=True,
    # add_eos_token=True,
)
# tokenizer.pad_token = tokenizer.eos_token

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)
model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    quantization_config=bnb_config,
    device_map="cuda:0")
sample_input = "Where is the best place to get cloud GPUs?"
tokenized_sample_input = tokenizer(sample_input, return_tensors="pt", max_length=256)
tokenized_sample_input = tokenized_sample_input.to("cuda")
tokenized_sample_output = model.generate(
    **tokenized_sample_input,
    max_new_tokens=200,
)
sample_output = tokenizer.decode(tokenized_sample_output[0])
assert False, sample_output

train_dataset = load_dataset('json', data_files='llama/dataset/simple.jsonl', split='train')
eval_dataset = load_dataset('json', data_files='llama/dataset/simple.jsonl', split='train')
def generate_and_tokenize_prompt(sample):
    prompt = f"Question: {sample['input']} Answer: {sample['output']}"
    return tokenizer(
        prompt,
        truncation=True,
        max_length=512,
        padding="max_length",
        return_tensors="pt"
        )
tokenized_train_dataset = train_dataset.map(generate_and_tokenize_prompt)
tokenized_eval_dataset = eval_dataset.map(generate_and_tokenize_prompt)

