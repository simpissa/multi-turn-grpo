import textarena as ta
import nltk
from typing import Any, Tuple, Optional, Union
import torch
from trl.trainer.grpo_trainer import pad

class MultiTurnBatchSampler:
    def __init__(self, b1: list[Any], b2: list[Any]):
        assert len(b1) == len(b2)
        self.batch_size = len(b1)
        self.envs = (b1, b2)
        self.remaining = [None, None]
        self.completion_turn = 0
        self.prompt_turn = 0
        self.buffer = [None, None]
        self._reset_batch(0)
        self._reset_batch(1) 

    # def extract_action():
    #     pass
    # def calculate_reward():
    #     pass

    def step(self,
            inputs: dict[Union[Any, torch.Tensor]],
            # completions_text: list[str], # ? 
            # prompt_ids: torch.Tensor, 
            # prompt_mask: torch.Tensor, 
            # completion_ids: torch.Tensor, 
            # completion_mask: torch.Tensor,
        ):
        completions_text, completion_ids, completion_mask = inputs["completions_text"], inputs["completion_ids"], inputs["completion_mask"]
        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]

        remaining = self.remaining[self.completion_turn % 2]        

        assert len(remaining) == len(completions_text) == prompt_ids.size(0) == prompt_mask.size(0) == completion_ids.size(0) == completion_mask.size(0)
        # print(f"{len(remaining)} {len(completions_text)} {prompt_ids.size(0)} {prompt_mask.size(0)} {completion_ids.size(0)} {completion_mask.size(0)}")
        envs = self.envs[self.completion_turn % 2]
        buffer = self.buffer[self.completion_turn % 2] 


        for index, completion, prompt_id, prompt_mask, completion_id, completion_mask in zip(
                list(remaining), completions_text, prompt_ids, prompt_mask, completion_ids, completion_mask):
            env = envs[index]

            action = self.extract_action(completion)

            buffer["prompt_ids"][index].append(prompt_id)
            buffer["prompt_mask"][index].append(prompt_mask)
            buffer["completion_ids"][index].append(completion_id)
            buffer["completion_mask"][index].append(completion_mask)

            done, info = env.step(action=action)
            if done:
                print(f"Batch {self.completion_turn % 2} Env {index} done")
                remaining.remove(index)
                # remaining.pop(index)

                reward, _ = env.close()
                reward = reward[0]
                buffer["rewards"][index] = self.calculate_reward(reward, len(buffer["prompt_ids"][index]))

        # padding
        for i in range(self.batch_size):
            if i not in remaining:
                buffer["prompt_ids"][i].append(torch.tensor([]))
                buffer["prompt_mask"][i].append(torch.tensor([]))
                buffer["completion_ids"][i].append(torch.tensor([]))
                buffer["completion_mask"][i].append(torch.tensor([]))

        self.completion_turn += 1
            
    def sample(self):
        envs = self.envs[self.prompt_turn % 2]
        remaining = self.remaining[self.prompt_turn % 2]

        observations = []
        for index in remaining:
            env = envs[index]
            _, observation = env.get_observation()
            observations.append(observation)
        self.prompt_turn += 1
        return observations

    def _reset_batch(self, batch_num: int):
        for env in self.envs[batch_num]:
            env.reset(num_players=1)
        self.remaining[batch_num] = [*range(self.batch_size)]
        self.buffer[batch_num] = {
            "prompt_ids" : [[] for _ in range(self.batch_size)],
            "prompt_mask" : [[] for _ in range(self.batch_size)],
            "completion_ids" : [[] for _ in range(self.batch_size)],
            "completion_mask" : [[] for _ in range(self.batch_size)],
            "rewards" : [[] for _ in range(self.batch_size)]
        }

    def prepare_backward(self):
        print("prepare_backward start")
        buffer = self.buffer[(self.completion_turn-1) % 2]
        self._reset_batch((self.completion_turn-1) % 2)

        for i in range(self.batch_size):
            buffer["prompt_ids"][i] = pad(buffer["prompt_ids"][i])
            buffer["prompt_mask"][i] = pad(buffer["prompt_mask"][i])
            buffer["completion_ids"][i] = pad(buffer["completion_ids"][i])
            buffer["completion_mask"][i] = pad(buffer["completion_mask"][i])

        buffer["prompt_ids"] = pad(buffer["prompt_ids"]).flatten(0, 1)
        buffer["prompt_mask"] = pad(buffer["prompt_mask"]).flatten(0, 1)
        buffer["completion_ids"] = pad(buffer["completion_ids"]).flatten(0, 1)
        buffer["completion_mask"] = pad(buffer["completion_mask"]).flatten(0, 1)

        buffer["rewards"] = pad(buffer["rewards"]) # (B, T)
        # how does this padding interact with padding of the other infos 
        print("prepare backward end")
        return buffer
    
    def batch_done(self):
       #return False if self.completion_turn == 0 else len(self.remaining[(self.completion_turn-1) % 2]) == 0
       return len(self.remaining[(self.completion_turn-1) % 2]) == 0
