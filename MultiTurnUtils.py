import torch

def create_batch(batch_size, group_size, game):

    from copy import deepcopy
    import textarena as ta
    import nltk
    nltk.download('averaged_perceptron_tagger_eng')

    envs = []
    for _ in range(batch_size // group_size):
        # env = ta.wrappers.LLMObservationWrapper(env=ta.make(game))
        env = ta.make(game)
        env.reset(num_players=1)
        for _ in range(group_size):
            envs.append(deepcopy(env))
    return envs

# filter action out of completion
def extract_action(completion: str):
    return completion
    
# given a single scalar reward and number of turns T, return a (1, T) tensor with rewards
def calculate_reward(reward: int, turns: int):
    return reward * torch.ones((1, turns))
