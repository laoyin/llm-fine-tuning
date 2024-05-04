import torch
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset
from prompt_setting import alpaca_prompt
from unsloth import FastLanguageModel


class llamaNN():

    def __init__(self):
        self.max_seq_length = 2048 # Choose any! We auto support RoPE Scaling internally!
        self.dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
        self.load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.

        # 4bit pre quantized models we support for 4x faster downloading + no OOMs.
        self.fourbit_models = [
            # "unsloth/mistral-7b-bnb-4bit",
            # "unsloth/mistral-7b-instruct-v0.2-bnb-4bit",
            # "unsloth/llama-2-7b-bnb-4bit",
            # "unsloth/gemma-7b-bnb-4bit",
            # "unsloth/gemma-7b-it-bnb-4bit", # Instruct version of Gemma 7b
            # "unsloth/gemma-2b-bnb-4bit",
            # "unsloth/gemma-2b-it-bnb-4bit", # Instruct version of Gemma 2b
            "unsloth/llama-3-8b-bnb-4bit", # [NEW] 15 Trillion token Llama-3
        ] # More models at https://huggingface.co/unsloth

        model, self.tokenizer = FastLanguageModel.from_pretrained(
            #model_name = "unsloth/llama-3-8b-bnb-4bit",
            model_name="models",
            max_seq_length = self.max_seq_length,
            dtype = self.dtype,
            load_in_4bit = self.load_in_4bit,
            # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
        )


        self.model = FastLanguageModel.get_peft_model(
            model,
            r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                              "gate_proj", "up_proj", "down_proj",],
            lora_alpha = 16,
            lora_dropout = 0, # Supports any, but = 0 is optimized
            bias = "none",    # Supports any, but = "none" is optimized
            # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
            use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
            random_state = 3407,
            use_rslora = False,  # We support rank stabilized LoRA
            loftq_config = None, # And LoftQ
        )

        self.EOS_TOKEN = self.tokenizer.eos_token # Must add EOS_TOKEN



    def get_dataset(self):
        EOS_TOKEN = self.EOS_TOKEN
        def formatting_prompts_func(examples):
            instructions = examples["instruction"]
            inputs = examples["input"]
            outputs = examples["output"]
            texts = []
            for instruction, input, output in zip(instructions, inputs, outputs):
                # Must add EOS_TOKEN, otherwise your generation will go on forever!
                text = alpaca_prompt.format(instruction, input, output) + EOS_TOKEN
                texts.append(text)
            return {"text": texts, }

        dataset = load_dataset("traning_datasets", split="train")
        dataset = dataset.map(formatting_prompts_func, batched=True, )
        return dataset


    def init_trainer(self):

        trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=self.get_dataset(),
            dataset_text_field="text",
            max_seq_length=self.max_seq_length,
            dataset_num_proc=2,
            packing=False,  # Can make training 5x faster for short sequences.
            args=TrainingArguments(
                per_device_train_batch_size=2,
                gradient_accumulation_steps=4,  # per_device_train_batch_size * gradient_accumulation_steps
                warmup_steps=5,
                max_steps=10,
                learning_rate=2e-4,
                fp16=not torch.cuda.is_bf16_supported(),
                bf16=torch.cuda.is_bf16_supported(),
                logging_steps=1,
                optim="adamw_8bit",
                weight_decay=0.01,
                lr_scheduler_type="linear",
                seed=3407,
                output_dir="outputs",
            ),
        )

        gpu_stats = torch.cuda.get_device_properties(0)
        start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
        max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
        print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
        print(f"{start_gpu_memory} GB of memory reserved.")

        trainer_stats = trainer.train()

        used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
        used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
        used_percentage = round(used_memory / max_memory * 100, 3)
        lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
        print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
        print(f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training.")
        print(f"Peak reserved memory = {used_memory} GB.")
        print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
        print(f"Peak reserved memory % of max memory = {used_percentage} %.")
        print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")


        self.tokenizer.save_pretrained("./lm3_lora_model")
        print("save pretrained model")
        self.model.save_pretrained("./lm3_lora_model")  # Local saving
        print("save lora model")
        #self.tokenizer.save_pretrained("lm3_lora_model")
        # model.push_to_hub("your_name/lora_model", token = "...") # Online saving
        # tokenizer.push_to_hub("your_name/lora_model", token = "...") # Online saving



    def inference(self):

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name="lm3_lora_model",  # YOUR MODEL YOU USED FOR TRAINING
            max_seq_length=self.max_seq_length,
            dtype=self.dtype,
            load_in_4bit=self.load_in_4bit,
        )
        FastLanguageModel.for_inference(model)  # Enable native 2x faster inference

        # alpaca_prompt = You MUST copy from above!

        inputs = tokenizer(
            [
                alpaca_prompt.format(
                    "中国的首都是什么?",  # instruction
                    "",  # input
                    "",  # output - leave this blank for generation!
                )
            ], return_tensors="pt").to("cuda")

        outputs = model.generate(**inputs, max_new_tokens=64, use_cache=True)
        print(tokenizer.batch_decode(outputs))
        #print("yxp out", outputs)