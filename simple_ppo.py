from env.balancebot_env import BalancebotEnv
from stable_baselines import PPO2
from stable_baselines.common.policies import FeedForwardPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines.bench import Monitor

import os
import time


log_dir = "/tmp/gym/{}".format(int(time.time()))
os.makedirs(log_dir, exist_ok=True)


# Create the environment
def make_env(rank):
    def _init():
        env = BalancebotEnv(render=False)
        env = Monitor(env, os.path.join(log_dir, str(rank)))
        return env
    return _init


#env = BalancebotEnv(render=False)
#env = Monitor(env, os.path.join(log_dir, str(0)))

# Create the RL Agwnt
class CustomPolicy(FeedForwardPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomPolicy, self).__init__(*args, **kwargs,
                                           layers=[32, 16],
                                           feature_extraction="mlp")

def main():
    num_cpu = 2 
    env = SubprocVecEnv([make_env(rank=i) for i in range(num_cpu)])
    model = PPO2(CustomPolicy, env, verbose=1)

    # Train and Save the agent
    model.learn(total_timesteps=1e5)
    model.save("ppo_save")

    # delete trained model to demonstrate loading
    del model 

    # evaluation
    env = DummyVecEnv([lambda: BalancebotEnv(render=False)])

    # Load the trained agent
    model = PPO2.load("ppo_save", env=env, policy=CustomPolicy)

    # Enjoy trained agent
    for ep in range(10):
        obs = env.reset()
        dones = False
        while not dones:
            action, _states = model.predict(obs)
            obs, rewards, dones, info = env.step(action)

if __name__ == '__main__':
    main()
