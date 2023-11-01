import gym
from gym import spaces
import mysql.connector
import numpy as np
import pandas as pd
from decouple import config

class TradingEnv(gym.Env):
    
    def __init__(self, time_penalty=-0.01, max_steps=10, time_threshold=10):
        super(TradingEnv, self).__init__()

        self.columns_to_keep = [
            'opening_price', 'closing_price', 'high_price', 'low_price',
            'price_change', 'total_volume', 'volume_weighted_avg_price', 
            'number_of_trades', 'buy_volume', 'sell_volume',
            'simple_moving_average', 'exponential_moving_average',
            'rsi', 'macd', 'market_trend'
        ]
        
        
        self.max_steps = max_steps
        self.current_step = 0
        self.initial_btc_balance = 0.003
        self.btc_balance = self.initial_btc_balance
        self.zar_balance = 0.0 
        self.total_profit = 0.0
        self.time_penalty = time_penalty
        self.time_threshold = time_threshold  
        self.open_orders = []
        
        
        self.action_space = spaces.Discrete(3)  # 0: Hold, 1: Open Long, 2: Close Long
        self.observation_space = spaces.Box(
            low=0, high=np.inf, shape=(self.max_steps, len(self.columns_to_keep)), dtype=np.float32
        )
        

        self.db_config = {
            "user": config("DB_USER"),
            "password": config("DB_PASSWORD"),
            "host": config("DB_HOST"),
            "port": config("DB_PORT"),
            "database": config("DB_NAME")
        }
        
        self.data = self.load_data()
        self.data = self.data[self.columns_to_keep]

    def load_data(self):
        print("Loading data...")
        connection = mysql.connector.connect(**self.db_config)
        cursor = connection.cursor()
        query = "SELECT * FROM tradehistory_preprocessed_1hour ORDER BY interval_start ASC"
        cursor.execute(query)
        data = cursor.fetchall()
        connection.close()
        print(f"Loaded {len(data)} rows of data.")
        return pd.DataFrame(data, columns=[column[0] for column in cursor.description])

    def reset(self):
        self.current_step = 0
        self.open_orders = []
        self.total_profit = 0.0  # Resetting the total_profit
        obs = self.data.iloc[self.current_step:self.current_step + self.max_steps].values.astype(np.float32)
        return obs
    
    def calculate_reward(self, action, current_close_price, current_market_trend):
        reward = 0.0
        ALPHA = 1.0
        BETA = 0.1
        GAMMA = 0.5
        DELTA = 0.2
        EPSILON = 0.05
        
        # Sample logic for calculating trade_profit_loss
        trade_amount_btc = 0.001
        trade_profit_loss = trade_amount_btc * (current_close_price - self.last_trade_price) if self.last_trade_price else 0.0
        reward += ALPHA * trade_profit_loss
        
        # Sample logic for calculating transaction cost
        transaction_cost = 0.001 * current_close_price * trade_amount_btc
        reward -= BETA * transaction_cost
        
        # Sample logic for calculating trend_alignment
        trend_alignment = 0
        if action == 1 and current_market_trend == 1:
            trend_alignment = 1
        elif action == 2 and current_market_trend == 0:
            trend_alignment = 1
        elif action != 0:
            trend_alignment = -1
        reward += GAMMA * trend_alignment
        
        # Sample logic for calculating risk_penalty
        stop_loss = 0.95 * self.last_trade_price
        take_profit = 1.05 * self.last_trade_price
        risk_penalty = 0.0
        if current_close_price <= stop_loss or current_close_price >= take_profit:
            risk_penalty = 0.02 * trade_amount_btc * current_close_price
        reward -= DELTA * risk_penalty
        
        # Time-sensitive penalty remains the same
        time_penalty = EPSILON * self.current_step / self.max_steps
        reward -= time_penalty
        
        return reward


    def step(self, action):
        self.current_step += 1
        current_data = self.data.iloc[self.current_step]
        current_close_price = float(current_data['closing_price'])
        current_market_trend = current_data['market_trend']

        reward = 0.0
        trade_profit_loss = 0.0

        # Calculate the reward based on the current state and action
        reward = self.calculate_reward(action, current_close_price, current_market_trend)
        
        # Update the state (observation space)
        new_obs = self.data.iloc[self.current_step:self.current_step + self.max_steps].values.astype(np.float32)
        

        # Check if the episode is done
        done = False
        if self.current_step >= len(self.data) - self.max_steps:
            done = True

        # Additional info, can be empty
        info = {"closing": current_close_price}

        return new_obs, reward, done, info

    def open_short(self, current_close_price):
        pass

    def close_short(self, current_close_price):
        pass

    
    def check_stop_loss(self, order, current_close_price):
        pass


    def should_take_profit(self, order, current_close_price):
        pass


    
    def total_asset_in_zar(self):
        current_close_price = float(self.data.iloc[self.current_step]['closing_price'])
        btc_in_zar = self.btc_balance * current_close_price
        return self.zar_balance + btc_in_zar
    
    
    def total_asset_in_btc(self):
        current_close_price = float(self.data.iloc[self.current_step]['closing_price'])
        zar_in_btc = self.zar_balance / current_close_price
        return self.btc_balance + zar_in_btc
    
    
    def log_metrics(self, episode, total_reward):
        print(f"Episode: {episode}, Total Reward: {total_reward}")
        print(f"Total Asset in ZAR: {self.total_asset_in_zar()}")
        print(f"Total Asset in BTC: {self.total_asset_in_btc()}")

    def render(self, mode='human', episode=None):
        if mode == 'human':
            print(f"Episode: {episode}, Step: {self.current_step}, ZAR Balance: {self.zar_balance}, BTC Balance: {self.btc_balance}")
            print(f"Total Asset in ZAR: {self.total_asset_in_zar()}")
            print(f"Total Asset in BTC: {self.total_asset_in_btc()}")






