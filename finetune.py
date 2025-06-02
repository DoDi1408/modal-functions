import modal

alpaca_prompt = """Below is an instruction that describes a task, paired with an incorrect input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""


HF_TOKEN= # DO MODAL SECRET
MINUTES = 60  # seconds
HOURS = 60 * MINUTES
N_GPU = 1

finetune_image = (
    modal.Image.from_registry("nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.12")
    .pip_install("cupy-cuda12x","torch","transformers>=4.51.0","unsloth",extra_index_url="https://flashinfer.ai/whl/cu124/torch2.5")
    .pip_install("datasets==3.6.0","huggingface_hub","hf_transfer")
    .apt_install("git")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})  # faster model transfers
)
hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)




app = modal.App("fine-tuning-some-model")

@app.function(
    image=finetune_image,
    gpu=f"A100-80GB:{N_GPU}",
    volumes={"/root/.cache/huggingface": hf_cache_vol},
    timeout=4 * HOURS,
)
def finetune():
    from unsloth import FastLanguageModel
    from trl import SFTTrainer, SFTConfig
    import torch
    import datasets


    print("Downloading model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/Qwen3-32B-unsloth-bnb-4bit",
        max_seq_length=4048,
        load_in_4bit=True,
        load_in_8bit=False,
        full_finetuning=False,
    )

    EOS_TOKEN = tokenizer.eos_token

    def formatting_prompts_func(dataset):
        instructions = dataset["instruction"]
        inputs = dataset["input"]
        outputs = dataset["output"]
        texts = []
        for instruction, input, output in zip(instructions, inputs, outputs):
            text = alpaca_prompt.format(instruction, input, output) + EOS_TOKEN
            texts.append(text)
        return {"text": texts, }

    print("Downloading and splitting datasets...")
    original_dataset = datasets.load_dataset("Romoamigo/oop-bad-code-to-good-code-cpp")

    formatted_dataset = original_dataset.map(formatting_prompts_func, batched=True)
    train_test_split_dataset = formatted_dataset["train"].train_test_split(test_size=0.15)

    train_and_test_AGAIN = train_test_split_dataset["train"].train_test_split(test_size=0.15)

    # train, test and eval
    test_dataset = train_test_split_dataset["test"]
    train_dataset = train_and_test_AGAIN["train"]
    eval_dataset = train_and_test_AGAIN["test"]


    model = FastLanguageModel.get_peft_model(
        model,
        r=32,  # Choose any number > 0! Suggested 8, 16, 32, 64, 128
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj", ],
        lora_alpha=32,  # Best to choose alpha = rank or rank*2
        lora_dropout=0,  # Supports any, but = 0 is optimized
        bias="none",  # Supports any, but = "none" is optimized
        # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
        use_gradient_checkpointing="unsloth",  # True or "unsloth" for very long context
        random_state=3407,
        use_rslora=True,  # We support rank stabilized LoRA
        loftq_config=None,  # And LoftQ
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=SFTConfig(
            dataset_text_field="text",
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            warmup_steps=20,
            #num_train_epochs=1,
            max_steps = 30, # lets try one first
            learning_rate=2e-4,
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            report_to="none",
        ),
    )
    trainer_stats = trainer.train()
    model.push_to_hub_merged("Romoamigo/Qwen3-32B-small-test", tokenizer, save_method="merged_16bit",token=HF_TOKEN)


