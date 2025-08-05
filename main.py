from unsloth import FastLanguageModel

from typing import Optional
from EnvWorker import EnvWorker
from datasets import Dataset
import torch
import torch.multiprocessing as mp
from torch.multiprocessing import Queue
from multiprocessing import freeze_support

from MultiTurnUtils import create_batch, extract_action, calculate_reward

from trl import GRPOConfig

from MultiTurnGRPOTrainer import MultiTurnGRPOTrainer

from torch.utils.data import SequentialSampler

from torch.utils.data import DataLoader

if __name__ == "__main__":
    freeze_support()
    max_seq_length = 1024

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "unsloth/Qwen3-4B",
        max_seq_length = max_seq_length,
        load_in_4bit = False, # False for LoRA 16bit
        fast_inference = False, # Enable vLLM fast inference
        max_lora_rank = 8,
        gpu_memory_utilization = 0.7, # Reduce if out of memory
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r = 8, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
        target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha = 8*2, # *2 speeds up training
        use_gradient_checkpointing = "unsloth", # Reduces memory usage
        random_state = 3407,
    )


    worker = EnvWorker(extract_action, calculate_reward)
    worker.create_batch = create_batch

    batch_size = 2
    group_size = 2
    steps = 10
    game = "Wordle-v0"

    system_prompt = "You are playing Wordle."

    ctx = mp.get_context("spawn")

    prompt_q = ctx.Queue()
    completion_q = ctx.Queue()
    training_q = ctx.Queue()

    p = ctx.Process(target=worker, args=(game, batch_size, group_size, steps, prompt_q, completion_q, training_q))
    p.start()

    def update_prompt(batch):
        observations = prompt_q.get()
        new_prompts = [
            [{"role": "system", "content": system_prompt},
            {"role": "user",   "content": observation}] for observation in observations
        ]

        return {
            'prompt': new_prompts,
        }

    ds = Dataset.from_dict({
        "dummy": [0]
    })
    ds.set_transform(update_prompt)

    max_prompt_length = 256
    steps = 10


    training_args = GRPOConfig(
        learning_rate = 5e-6,
        adam_beta1 = 0.9,
        adam_beta2 = 0.99,
        weight_decay = 0.1,
        warmup_ratio = 0.1,
        lr_scheduler_type = "cosine",
        optim = "adamw_torch_fused",
        logging_steps = 1,
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 1,
        num_generations = 2,
        max_prompt_length = max_prompt_length,
        max_completion_length = max_seq_length - max_prompt_length,
        # num_train_epochs = 1,
        max_steps = steps,
        save_steps = 10,
        max_grad_norm = 0.1,
        report_to = "none",
        output_dir = "outputs",
    )
    training_args.per_device_train_batch_size = 1
    training_args.num_generations = 1
    training_args.steps_per_generation = 1

    trainer = MultiTurnGRPOTrainer(
        group_size = 2,
        model = model,
        processing_class = tokenizer,
        args = training_args,
        train_dataset = ds,
        completion_q=completion_q,
        training_q=training_q,
        reward_funcs = []
    )

    class InfiniteSampler(SequentialSampler):
        def __init__(self, steps, **kwargs):
            self.steps = steps
            super().__init__(**kwargs)

        def __iter__(self):
            while True:
                # yield from range(len(self.data_source))
                yield 0

        def __len__(self):
            return self.steps

    def _get_train_sampler(self, dataset: Optional[Dataset] = None):
        if dataset is None:
            dataset = self.train_dataset
        return InfiniteSampler(steps, data_source=dataset)
        # return SequentialSampler(dataset)

    MultiTurnGRPOTrainer._get_train_sampler = _get_train_sampler

    def get_train_dataloader(self):
        # if self.train_dataset is None:
        #     raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        data_collator = self.data_collator
        # if is_datasets_available() and isinstance(train_dataset, datasets.Dataset):
        #     train_dataset = self._remove_unused_columns(train_dataset, description="training")
        # else:
        #     data_collator = self._get_collator_with_removed_columns(data_collator, description="training")

        dataloader_params = {
            "batch_size": self._train_batch_size * self.args.steps_per_generation,  # < this is the change
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }

        dataloader_params["sampler"] = self._get_train_sampler()
        dataloader_params["drop_last"] = self.args.dataloader_drop_last
        # if version.parse(transformers.__version__) >= version.parse("4.52.0"):
        #     # from transformers 4.52.0, the `seed_worker` requires the `num_workers` and `rank` arguments
        #     dataloader_params["worker_init_fn"] = partial(
        #         seed_worker, num_workers=self.args.dataloader_num_workers, rank=self.args.process_index
        #     )
        # else:
        #     dataloader_params["worker_init_fn"] = seed_worker
        dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor

        return DataLoader(train_dataset, **dataloader_params)

    MultiTurnGRPOTrainer.get_train_dataloader = get_train_dataloader

    torch._dynamo.config.cache_size_limit = 32

    trainer.train()