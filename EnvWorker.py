from torch.multiprocessing import Queue
from typing import Callable

class EnvWorker():
    def __init__(self, extract_action: Callable, calculate_reward: Callable):
        self.extract_action = extract_action
        self.calculate_reward = calculate_reward

    # def create_batch():
    #     pass
    
    def __call__(self, game: str, batch_size: int, group_size: int, steps: int, prompt_q: Queue, completion_q: Queue, training_q: Queue):
        # prompt_q  sampler --> dataloader(main)
        # completion_q  generator(main) --> sampler
        # training_q    sampler --> trainer(main)

        from MultiTurnBatchSampler import MultiTurnBatchSampler
        b1 = self.create_batch(batch_size=batch_size, group_size=group_size, game=game)
        b2 = self.create_batch(batch_size=batch_size, group_size=group_size, game=game)
        sampler = MultiTurnBatchSampler(b1, b2)
        sampler.extract_action = self.extract_action
        sampler.calculate_reward = self.calculate_reward

        for _ in range(steps):
            # cold start
            x = sampler.sample()
            prompt_q.put(x)
            # prompt_q.put(sampler.sample())
            # print(f"Observation: {x}")
            while not sampler.batch_done():
                observations = sampler.sample()
                prompt_q.put(observations)
                # print(f"Observation: {observations}")

                completions = completion_q.get()
                # print(f"Completion: {completions["completions_text"]}")
                sampler.step(completions)

            training_q.put(sampler.prepare_backward()) 

            # if not completion_q.empty(): # guard since prepare_backward should be faster than generate or batches finish at the same time
            completions = completion_q.get()
            # print(f"Completion: {completions["completions_text"]}")
            sampler.step(completions) # this assumes prepare_backward is faster (?)

            # print(f"Prompt Turn: {sampler.prompt_turn}")
            # print(f"Completion Turn: {sampler.completion_turn}")
            assert sampler.prompt_turn == sampler.completion_turn, "generate is slower than prepare_backward"