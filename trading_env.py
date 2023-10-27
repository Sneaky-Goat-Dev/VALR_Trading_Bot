import gym
from gym import spaces
import mysql.connector
import numpy as np
import pandas as pd
from decouple import config
import matplotlib.pyplot as plt



class TradingEnv(gym.Env):
    
    def __init__(self):
        super(TradingEnv, self).__init__()

        self.action_space = spaces.Discrete(3)  # Buy, sell, hold
        self.observation_space = spaces.Box(
            low=0, high=np.inf, shape=(5, 17), dtype=np.float32
        )  # 5 hours of data, each with 17 features

        self.db_config = {
            "user": config("DB_USER"),
            "password": config("DB_PASSWORD"),
            "host": config("DB_HOST"),
            "port": config("DB_PORT"),
            "database": config("DB_NAME")
        }
        self.connection = None
        self.cursor = None

        self.current_step = 0
        self.data = self.load_data()
        
        self.actions_taken = []  # List to store the actions taken during an episode
        
    
    def load_data(self):
        print("Loading data...")
        self.connection = mysql.connector.connect(**self.db_config)
        self.cursor = self.connection.cursor()
        query = "SELECT * FROM tradehistory_preprocessed_1hour ORDER BY interval_start ASC"
        self.cursor.execute(query)
        data = self.cursor.fetchall()
        self.connection.close()
        print(f"Loaded {len(data)} rows of data.")
        return pd.DataFrame(data, columns=[column[0] for column in self.cursor.description])


    def reset(self):
        self.current_step = 0
        self.actions_taken = []  # Reset the actions list
        return self.data.iloc[self.current_step:self.current_step + 5].values


    def step(self, action):
        print(f"Taking action {action} at step {self.current_step}")
        self.current_step += 1
        self.actions_taken.append(action)  # Store the action taken

        # Inside the step method
        if self.current_step + 5 > len(self.data) - 1 or self.current_step >= 100:
            done = True
        else:
            done = False

        prev_state = self.data.iloc[self.current_step - 1:self.current_step + 4]
        current_state = self.data.iloc[self.current_step:self.current_step + 5]

        # Assume a simple reward based on the price change
        reward = current_state['closing_price'].iloc[-1] - prev_state['closing_price'].iloc[-1]

        if action == 0:  # Buy
            # Your buying logic here
            pass
        elif action == 1:  # Sell
            # Your selling logic here
            pass
        # action == 2 is Hold, do nothing

        return current_state.values, reward, done, {}


    def render(self, mode='human'):
        if self.current_step < 5:
            print("Not enough data to render. Continue the episode.")
            return
        
        window_data = self.data.iloc[self.current_step - 5:self.current_step]
        actions_window = self.actions_taken[self.current_step - 5:self.current_step]  # Get the actions for the current window
        
        plt.figure(figsize=(10, 6))

        # Plot the closing prices
        plt.plot(range(len(window_data)), window_data['closing_price'], marker='o')

        # Annotate the buy/sell/hold actions
        for i, (index, row) in enumerate(window_data.iterrows()):
            action = actions_window[i]
            if action == 0:  # Buy
                plt.annotate('Buy', (i, row['closing_price']), textcoords="offset points", xytext=(0,10), ha='center', color='g')
            elif action == 1:  # Sell
                plt.annotate('Sell', (i, row['closing_price']), textcoords="offset points", xytext=(0,10), ha='center', color='r')
            # Hold action is not annotated

        plt.title('Trading actions over the last 5 steps')
        plt.xlabel('Step')
        plt.ylabel('Closing Price')
        plt.grid()
        plt.show()





env = TradingEnv()
observation = env.reset()
for _ in range(100):
    action = env.action_space.sample()
    print(f"Sampling action: {action}")
    observation, reward, done, info = env.step(action)
    if done:
        print("Episode done, rendering...")
        env.render()
        observation = env.reset()
