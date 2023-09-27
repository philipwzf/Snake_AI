import gym
import numpy as np
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import PPO
from snake_game import SnakeGame
import random
import os

class SnakeWrapper(gym.Env):

    def __init__(self, seed,board_size,silent_mode):
        super().__init__()
        self.game = SnakeGame(seed=seed,board_size=board_size,silent_mode=silent_mode)

        self.action_space = gym.spaces.Discrete(4)
        self.observation_space = gym.spaces.Box(
            low=0, high=255,
            shape=(48, 48, 3),
            dtype=np.uint8
        )
        
        self.max_steps = 10000
        # Counter of steps per episode
        self.current_step = 0
        self.max_growth = self.game.board_size**2 - len(self.game.snake)

    def reset(self):
        """
        Reset the environment
        """
        # Reset the counter
        self.game.reset()
        self.current_step = 0
        self.max_growth = self.game.board_size**2 - len(self.game.snake)

        # Generate obs for the agent to learn
        obs = self.generate_obs()

        return obs

    def render(self):
        self.game.render()


    def step(self, action):

        done, info = self.game.step(action)
        self.current_step += 1
        reward = len(self.game.snake)
        #the new head of the snake
        row, col = self.game.snake[0]
        
        if len(self.game.snake)>=self.game.board_size**2:
            # define a point for victory
            vic = 10000
            reward += vic
        elif ((row, col) in self.game.snake)\
                or row < 0\
                or col < 0\
                or row >= self.game.board_size\
                or col >= self.game.board_size:
            # the shorter when die -> more points deducted
            reward -= self.max_growth 
        # When the number of steps reaches the maximum
        elif self.current_step >= self.max_steps:
            reward -= self.max_growth 

        if(info["food_obtained"]):
            reward += self.max_growth 
            self.max_growth -= 1

        # Check if the head is closer to food now
        food_row,food_col = self.game.food
        prev_row,prev_col = self.game.snake[1]
        curr_dis_to_food = np.abs(food_row-row)+np.abs(food_col-col)
        prev_dis_to_food = np.abs(food_row-prev_row)+np.abs(food_col-prev_col)
        if (curr_dis_to_food > prev_dis_to_food):
            reward += 1/curr_dis_to_food
        else:
            reward -= 1/curr_dis_to_food

        # Generate obs for the agent to learn
        obs = self.generate_obs()

        return obs, reward, done, info

    def generate_obs(self):
        obs = np.zeros((12,12,3),dtype=np.uint8)
        color_list = np.linspace(255, 100, len(self.game.snake), dtype=np.uint8)
        
        # mark the snake
        for i in range(len(self.game.snake)):
            obs[self.game.snake[i]] = (0,0,color_list[i])
        
        # mark the food
        obs[self.game.food] = (255,0,0)

        obs = np.repeat(np.repeat(obs, 4, axis=0), 4, axis=1)
        return obs

def check_snakeWrapper(env):
    # If the environment don't follow the interface, an error will be thrown
    check_env(env, warn=True)

if __name__ == "__main__":
    LOG_DIR = "logs"

    os.makedirs(LOG_DIR, exist_ok=True)

    seed = random.randint(0, 1e9)   
    env = SnakeWrapper(seed=seed, board_size=12,silent_mode=False)
    env = Monitor(env)
    model = PPO(
            "CnnPolicy",
            env,
            device="cuda",
            verbose=1,
            n_steps=2048,
            batch_size=512,
            n_epochs=4,
            gamma=0.94,
            tensorboard_log=LOG_DIR
        )
    model.learn(10000)