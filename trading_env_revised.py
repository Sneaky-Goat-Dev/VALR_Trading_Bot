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
        self.btc_balance = 0.0
        self.zar_balance = 2000.0
        self.total_profit = 0.0
        self.time_penalty = time_penalty
        self.time_threshold = time_threshold  # The maximum number of steps a trade can be open without penalty
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
    
    def calculate_reward(self, trade_profit_loss, existing_reward=0.0):
        reward = existing_reward
        profit_multiplier = 2.0  # Encourage profit
        loss_penalty = 1.5  # Discourage loss

        if trade_profit_loss > 0:
            reward += trade_profit_loss * profit_multiplier
        else:
            reward -= abs(trade_profit_loss) * loss_penalty

        return reward

    def step(self, action):
        self.current_step += 1
        current_data = self.data.iloc[self.current_step]
        current_close_price = current_data['closing_price']
        current_market_trend = current_data['market_trend']  # Get the market_trend value from the current data

        # Initialize reward and trade_profit_loss
        reward = 0.0
        trade_profit_loss = 0.0

        for order in self.open_orders:
            order['time_open'] += 1  # Increment the time_open for each order
            
            # Check for stop-loss or time penalty
            if current_close_price <= order['stop_loss'] or order['time_open'] > self.time_threshold:
                trade_profit_loss += self.close_long(order, current_close_price)
                if order['time_open'] > self.time_threshold:
                    # print('Time penalty applied.')
                    reward += self.time_penalty  # Apply time penalty
            # Check for 5% profit to close the trade immediately
            if self.should_take_profit(order, current_close_price):
                # print("5% profit reached. Closing trade.")
                trade_profit_loss += self.close_long(order, current_close_price)

        if action == 0:  # Hold
            if current_market_trend == 1:  # If it's an uptrend
                reward += 0.1  # Reward for holding during an uptrend
        elif action == 1:  # Open Long
            self.open_long(current_close_price)
        elif action == 2:  # Close Long
            if self.open_orders:
                trade_profit_loss += self.close_long(self.open_orders[0], current_close_price)

        # Calculate reward
        reward = self.calculate_reward(trade_profit_loss, reward)

        # Update the state (observation space)
        new_obs = self.data.iloc[self.current_step:self.current_step + self.max_steps].values.astype(np.float32)

        # Check if the episode is done
        done = False
        if self.current_step >= len(self.data) - self.max_steps:
            done = True

        # Additional info, can be empty
        info = {"closing": current_close_price}

        return new_obs, reward, done, info

    def open_long(self, current_close_price):
        current_close_price = float(current_close_price)  # Convert to float
        amount_to_buy = (self.zar_balance / 100) / current_close_price
        # Check if there are enough ZAR funds to open the long order
        if amount_to_buy * current_close_price > self.zar_balance:
            print("Not enough ZAR to open long order.")
            return  # exit the function
        
        self.btc_balance += amount_to_buy
        self.zar_balance -= self.zar_balance / 100
        stop_loss = current_close_price * 0.99  # 5% stop-loss
        total_asset_at_order_time = self.total_asset_in_zar()
        self.open_orders.append({
            'type': 'long',
            'close_price': current_close_price,
            'amount': amount_to_buy,
            'stop_loss': stop_loss,
            'time_open': 0,
            'total_asset_at_order_time': total_asset_at_order_time  # New key-value pair
        })

    def close_long(self, order, current_close_price):
        current_close_price = float(current_close_price)  # Convert to float
        # Check if there are enough BTC to close the long order
        if order['amount'] > self.btc_balance:
            # print("Not enough BTC to close long order.")
            return 0  # return 0 as there's no profit or loss
        
        sell_price = order['amount'] * current_close_price
        bought_at = order['amount'] * order['close_price']
        profit = sell_price - bought_at
        self.total_profit += profit  # Update the total_profit here
        self.zar_balance += sell_price
        self.btc_balance -= order['amount']
        self.open_orders.remove(order)
        return profit
    
    
    def should_take_profit(self, order, current_close_price):
        current_close_price = float(current_close_price)
        sell_price = order['amount'] * current_close_price
        bought_at = order['amount'] * order['close_price']
        profit = sell_price - bought_at
        profit_percent = (profit / order['total_asset_at_order_time']) * 100  # Calculate profit percentage

        return profit_percent >= 5.0


    
    def total_asset_in_zar(self):
        current_close_price = float(self.data.iloc[self.current_step]['closing_price'])
        btc_in_zar = self.btc_balance * current_close_price
        return self.zar_balance + btc_in_zar
    
    
    def log_metrics(self, episode, total_reward):
        print(f"Episode: {episode}, Total Reward: {total_reward}")
        print(f"Total Asset in ZAR: {self.total_asset_in_zar()}")

    def render(self, mode='human', episode=None):
        if mode == 'human':
            print(f"Episode: {episode}, Step: {self.current_step}, ZAR Balance: {self.zar_balance}, BTC Balance: {self.btc_balance}")
            print(f"Total Asset in ZAR: {self.total_asset_in_zar()}")






