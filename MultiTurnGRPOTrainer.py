from trl import GRPOTrainer
import torch
import torch.nn as nn
from typing import *
from MultiTurnBatchSampler import MultiTurnBatchSampler
import torch.multiprocessing as mp

from trl.trainer.grpo_trainer import maybe_apply_chat_template, unwrap_model_for_generation, FSDP, is_conversational, profiling_context, apply_chat_template, warnings, gather, pad, gather_object, nanstd
from contextlib import nullcontext

class MultiTurnGRPOTrainer(GRPOTrainer):
    def __init__(self, group_size, completion_q, training_q, **kwargs):
        self.group_size = group_size
        self.completion_q = completion_q
        self.training_q = training_q
        super().__init__(**kwargs)

    def _generate_and_score_completions(
        self, inputs: list[dict[str, Union[torch.Tensor, Any]]]
    ) -> dict[str, Union[torch.Tensor, Any]]:

        device = self.accelerator.device
        mode = "train" if self.model.training else "eval"

        temp_iterator = iter(self.get_train_dataloader())

        first_iter = True
        while self.training_q.empty() or first_iter:
            if not first_iter:
                # inputs = next(temp_iterator).to(self.args.device)
                inputs = self.get_batch_samples(temp_iterator, 1, self.args.device)
                # print(f"BATCH: {inputs}")
                inputs = inputs[0][0]
                # print("NEXT ITER")
            first_iter = False

            prompts = [x["prompt"] for x in inputs]
            
            prompts_text = [maybe_apply_chat_template(example, self.processing_class)["prompt"] for example in inputs]
            prompt_inputs = self.processing_class(
                text=prompts_text, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False
            )
            prompt_inputs = super(type(self).__mro__[-3], self)._prepare_inputs(prompt_inputs)
            prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]

            if self.max_prompt_length is not None:
                prompt_ids = prompt_ids[:, -self.max_prompt_length :]
                prompt_mask = prompt_mask[:, -self.max_prompt_length :]

            # generation
            with unwrap_model_for_generation(
                self.model_wrapped, self.accelerator, gather_deepspeed3_params=self.args.ds3_gather_for_generation
            ) as unwrapped_model:
                with (
                    FSDP.summon_full_params(self.model_wrapped, recurse=False)
                    if self.is_fsdp_enabled
                    else nullcontext()
                ):
                    prompt_completion_ids = unwrapped_model.generate(
                        prompt_ids, attention_mask=prompt_mask, generation_config=self.generation_config
                    )

            # Compute prompt length and extract completion ids
            prompt_length = prompt_ids.size(1)
            prompt_ids = prompt_completion_ids[:, :prompt_length]
            completion_ids = prompt_completion_ids[:, prompt_length:]
        
            # Mask everything after the first EOS token
            is_eos = completion_ids == self.processing_class.eos_token_id
            eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
            eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
            sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
            completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()
            
            # Convert tensor to a list of lists of token IDs. This will be passed to the reward function, avoiding the need
            # to re-tokenize completions if the reward is computed from tokens.
            completion_ids_list = [
                [id.item() for id, m in zip(row, mask_row) if m] for row, mask_row in zip(completion_ids, completion_mask)
            ]

            # Sum along sequence dimension (dim=1) to get completion length per sequence, used for logging
            # completion_lengths = completion_mask.sum(1)

            # If mask_truncated_completions is enabled, zero out truncated completions in completion_mask
            if self.mask_truncated_completions:
                truncated_completions = ~is_eos.any(dim=1)
                completion_mask = completion_mask * (~truncated_completions).unsqueeze(1).int()

            # Concatenate prompt_mask with completion_mask for logit computation
            attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)  # (B, P+C)

            logits_to_keep = completion_ids.size(1)  # we only need to compute the logits for the completion tokens
            batch_size = self.args.per_device_train_batch_size if mode == "train" else self.args.per_device_eval_batch_size

            with torch.no_grad():
                # When using num_iterations == 1 and steps_per_generation <= gradient_accumulation_steps
                # old_per_token_logps == per_token_logps, so we can skip it's computation here, and use
                # per_token_logps.detach() instead.
                if self.num_iterations > 1 or self.args.steps_per_generation > self.args.gradient_accumulation_steps:
                    old_per_token_logps = self._get_per_token_logps(
                        self.model, prompt_completion_ids, attention_mask, logits_to_keep, batch_size
                    )
                else:
                    old_per_token_logps = None

            # Decode the generated completions
            completions_text = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
            if is_conversational(inputs[0]):
                completions = []
                for prompt, completion in zip(prompts, completions_text):
                    bootstrap = prompt.pop()["content"] if prompt[-1]["role"] == "assistant" else ""
                    completions.append([{"role": "assistant", "content": bootstrap + completion}])
            else:
                completions = completions_text

            self.completion_q.put({
                "completions_text" : completions,
                "completion_ids" : completion_ids, 
                "completion_mask" : completion_mask,
                "prompt_ids" : prompt_ids, 
                "prompt_mask" : prompt_mask,
                "old_per_token_logps" : old_per_token_logps
            })

            # print()
            # print(f"Prompt: {prompts_text}")
            # print(f"Completion: {completions}")
            # print()

        inputs = self.training_q.get()

        # print("Training")

        rewards = inputs.pop("rewards").to(device)
        max_turns = rewards.shape[1]

        # Compute grouped-wise rewards
        mean_grouped_rewards = rewards.view(-1, self.group_size, max_turns).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, self.group_size, max_turns).std(dim=1)
        is_std_zero = torch.isclose(std_grouped_rewards, torch.zeros_like(std_grouped_rewards))

        # Normalize the rewards to compute the advantages
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.group_size, dim=0)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.group_size, dim=0)
        advantages = rewards - mean_grouped_rewards
        if self.scale_rewards:
            advantages = advantages / (std_grouped_rewards + 1e-4)

        # # Slice to keep only the local part of the data
        # process_slice = slice(
        #     self.accelerator.process_index * len(prompts),
        #     (self.accelerator.process_index + 1) * len(prompts),
        # )
        all_process_advantages = advantages.clone()  # keep the aggregated advantages for logging
        # advantages = advantages[process_slice, :]

        inputs["advantages"] = advantages.flatten()

        self._metrics[mode]["reward"].append(mean_grouped_rewards.mean().item())
        # Calculate mean reward per function, but only for samples where the function was applied (non-NaN values)
        # for i, reward_func_name in enumerate(self.reward_func_names):
        #     mean_rewards = torch.nanmean(rewards_per_func[:, i]).item()
        #     self._metrics[mode][f"rewards/{reward_func_name}/mean"].append(mean_rewards)
        #     std_rewards = nanstd(rewards_per_func[:, i]).item()
        #     self._metrics[mode][f"rewards/{reward_func_name}/std"].append(std_rewards)
        # self._metrics[mode]["reward"].append(mean_grouped_rewards.mean().item())
        # self._metrics[mode]["reward_std"].append(std_grouped_rewards.mean().item())
        # self._metrics[mode]["frac_reward_zero_std"].append(is_std_zero.float().mean().item())


        return inputs