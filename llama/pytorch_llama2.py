from datasets import load_dataset
import torch
import transformers
import peft
import os
from datetime import datetime
import wandb

MAX_INPUT, MAX_OUTPUT = 256, 256
# If False, just load the fine-tuned model from the tmp directory.
FINETUNE_MODEL = True


def get_model_and_tokenizer():
    base_model_id = "meta-llama/Llama-2-7b-chat-hf"
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        base_model_id,
        padding_side="left",  # More efficient for decoder only model
    )
    tokenizer.pad_token = tokenizer.eos_token

    bnb_config = transformers.BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    model = transformers.AutoModelForCausalLM.from_pretrained(
        base_model_id,
        quantization_config=bnb_config,
        device_map="cuda:0")
    return model, tokenizer
model, tokenizer = get_model_and_tokenizer()


MODEL_NAME = "Llama-2-7b-chat-hf"
PROJECT_NAME = "finetune"
STEPS = 50
output_dir = f"/tmp/toy_models/{MODEL_NAME}/{PROJECT_NAME}"
if FINETUNE_MODEL:
    def print_trainable_parameters(model, prefix=""):
        trainable_params = 0
        all_param = 0
        for _, param in model.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        print(prefix, model)
        print(
            f"{prefix} trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
        )
    print_trainable_parameters(model, "before peft")

    # Apply Lora
    model = peft.prepare_model_for_kbit_training(model)
    config = peft.LoraConfig(
        r=32,
        lora_alpha=64,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
            "lm_head",
        ],
        bias="none",
        lora_dropout=0.05,  # Conventional
        task_type="CAUSAL_LM",
    )
    model = peft.get_peft_model(model, config)
    print_trainable_parameters(model, "after peft")

    # Load dataset
    train_dataset = load_dataset(
        'json', data_files='llama/dataset/simple.jsonl', split='train')
    eval_dataset = load_dataset(
        'json', data_files='llama/dataset/simple.jsonl', split='train')

    def generate_and_tokenize_prompt(sample):
        prompt = f"Question: {sample['input']} Answer: {sample['output']}"
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=MAX_INPUT + MAX_OUTPUT,
            padding="max_length",
        )
        return result
    tokenized_train_dataset = train_dataset.map(generate_and_tokenize_prompt)
    tokenized_eval_dataset = eval_dataset.map(generate_and_tokenize_prompt)

    # Train
    os.makedirs(output_dir, exist_ok=True)
    wandb.init(project=f"{MODEL_NAME}-{PROJECT_NAME}", dir=output_dir)
    trainer = transformers.Trainer(
        model=model,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_eval_dataset,
        args=transformers.TrainingArguments(
            output_dir=output_dir,
            gradient_checkpointing=True,
            warmup_steps=1,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=1,
            max_steps=STEPS,
            learning_rate=2.5e-5,  # Want a small lr for finetuning
            bf16=False,                  # Enable if GPU >= Ampere
            optim="paged_adamw_8bit",
            # Directory for storing logs
            logging_dir=os.path.join(output_dir, "logs"),
            save_strategy="steps",       # Save the model checkpoint every logging step
            save_steps=10,                # Save checkpoints every 50 steps
            evaluation_strategy="steps",  # Evaluate the model every logging step
            eval_steps=10,               # Evaluate and save checkpoints every 50 steps
            do_eval=True,                # Perform evaluation at the end of training
            report_to="wandb",           # Comment this out if you don't want to use weights & baises
        ),
        data_collator=transformers.DataCollatorForLanguageModeling(
            tokenizer, mlm=False),
    )
    # silence the warnings. Please re-enable for inference!
    model.config.use_cache = False
    trainer.train(resume_from_checkpoint=False)

del (model)
del (tokenizer)

# Inference
def run_inference(model, tokenizer, tokenized_sample_input):
    with torch.no_grad():
        tokenized_sample_output = model.generate(
            **tokenized_sample_input,
            max_new_tokens=MAX_OUTPUT,
            pad_token_id=tokenizer.pad_token_id,
        )
        sample_output = tokenizer.decode(tokenized_sample_output[0])
    return sample_output
base_model, tokenizer = get_model_and_tokenizer()
sample_input = "Where is the best place to get cloud GPUs?"
tokenized_sample_input = tokenizer(
    sample_input, return_tensors="pt", max_length=MAX_INPUT, padding=True)
tokenized_sample_input = tokenized_sample_input.to("cuda")
print("Output before finetuning:\n", run_inference(base_model, tokenizer, tokenized_sample_input))

model = peft.PeftModel.from_pretrained(
    base_model, os.path.join(output_dir, f"checkpoint-{STEPS}"))
model.eval()
print("Output after finetuning:\n", run_inference(model, tokenizer, tokenized_sample_input))
