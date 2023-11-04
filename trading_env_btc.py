import gym
from gym import spaces
import mysql.connector
import numpy as np
import pandas as pd
from decouple import config

class TradingEnv(gym.Env):
    
    def __init__(self, time_penalty=-0.01, max_steps=10, time_threshold=10,
                 ALPHA=1.0,  # Weight for trade profit or loss
                 BETA=0.1,   # Weight for transaction cost
                 GAMMA=0.5,  # Weight for market trend alignment
                 DELTA=0.2,  # Weight for risk penalty (stop loss, take profit)
                 EPSILON=0.05,  # Weight for time penalty
                 ZETA=0.1  # Weight for take profit
                 ):
        super(TradingEnv, self).__init__()
        
        # Store the hyperparameters
        self.ALPHA = ALPHA
        self.BETA = BETA
        self.GAMMA = GAMMA
        self.DELTA = DELTA
        self.EPSILON = EPSILON
        self.ZETA = ZETA

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
        self.closing_prices = []  # List to store closing prices
        self.trades = []  # List to store trades
        
        
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
        
        # Check if the last action was successful
        last_order_successful = self.open_orders[-1]['successful'] if self.open_orders else True

        trade_profit_loss = self.calculate_trade_profit_loss(current_close_price)
        reward += self.ALPHA * trade_profit_loss

        transaction_cost = self.calculate_transaction_cost()
        reward -= self.BETA * transaction_cost

        trend_alignment = self.calculate_trend_alignment(action, current_market_trend)
        reward += self.GAMMA * trend_alignment

        risk_penalty = self.calculate_risk_penalty(current_close_price)
        reward -= self.DELTA * risk_penalty
        
        take_profit_reward = self.calculate_take_profit(current_close_price)
        reward += self.ZETA * take_profit_reward  # Adding reward for take profit

        time_penalty = self.calculate_time_penalty()
        reward -= self.EPSILON * time_penalty

        return reward


    def step(self, action):
        self.current_step += 1
        current_data = self.data.iloc[self.current_step]
        current_close_price = float(current_data['closing_price'])
        current_market_trend = current_data['market_trend']
        
        
        self.closing_prices.append(current_close_price)
        if action in [1, 2]:  # 1: Open Short, 2: Close Short
            trade = {
                'step': self.current_step,
                'price': current_close_price,
                'action': action
            }
            self.trades.append(trade)
        

        # Execute the action
        if action == 1:  # Open Short
            self.open_short(current_close_price)
        elif action == 2:  # Close Short
            self.close_short(current_close_price)

        # Calculate the reward based on the current state and action
        reward = self.calculate_reward(action, current_close_price, current_market_trend)

        # Update the state (observation space)
        new_obs = self.data.iloc[self.current_step:self.current_step + self.max_steps].values.astype(np.float32)

        # Check if the episode is done
        done = False
        if self.current_step >= len(self.data) - self.max_steps:
            done = True

        # New termination condition: check for a 20% loss of the initial account balance
        total_asset_in_btc = self.total_asset_in_btc()
        if total_asset_in_btc <= 0.8 * self.initial_btc_balance:
            print("Terminating episode due to a 20% loss of the initial account balance.")
            done = True

        # Additional info, can be empty
        info = {"closing": current_close_price}

        return new_obs, reward, done, info

    def open_short(self, current_close_price):
        # Calculate 1% of the total account balance in BTC
        risk_amount = 0.01 * self.btc_balance
        
        # Step 1: Check if you have enough BTC to sell for the short position.
        if risk_amount <= 0:
            # print("Insufficient BTC balance to open a new short position.")
            return

        # Step 2: Deduct the risk amount of BTC to be sold from your BTC balance.
        self.btc_balance -= risk_amount

        # Step 3: Add the equivalent ZAR to your ZAR balance.
        self.zar_balance += risk_amount * current_close_price

        # Step 4: Create a new order.
        order = {
            'entry_price': current_close_price,
            'amount': risk_amount,
            'successful': self.btc_balance >= risk_amount
        }

        # Step 5: Add this new order to the open_orders list.
        self.open_orders.append(order)

        # print(f"Opened a new short position of {risk_amount} BTC at {current_close_price} ZAR.")



    def close_short(self, current_close_price):
        if not self.open_orders:
            # print("No open orders to close.")
            return

        # Assuming last_order is the order you want to close
        last_order = self.open_orders[-1]

        # Calculate profit or loss
        profit_loss = (last_order['entry_price'] - current_close_price) * last_order['amount']

        # Update BTC and ZAR balances
        self.btc_balance += (self.zar_balance + profit_loss - self.calculate_transaction_cost()) / current_close_price

        # Reset ZAR balance since the short position is closed
        self.zar_balance = 0.0

        # Determine if the trade was successful and update the 'successful' key
        last_order['successful'] = profit_loss > 0

        # Remove the closed order from the open_orders list
        self.open_orders.pop()

        # print(f"Closed short position, Profit/Loss: {profit_loss}, Successful: {last_order['successful']}")
        
    
    def calculate_trade_profit_loss(self, current_close_price):
        if self.open_orders:
            last_order = self.open_orders[-1]
            return (last_order['entry_price'] - current_close_price) * last_order['amount']
        return 0.0


    def calculate_transaction_cost(self):
        if self.open_orders:
            last_order = self.open_orders[-1]
            return 0.01 * last_order['amount']  # 1% of the order amount in BTC
        return 0.0  # return zero if there are no open orders
    
    
    def calculate_trend_alignment(self, action, current_market_trend):
        trend_alignment = 0
        if action == 1 and current_market_trend == 0:  # Open short in bearish market
            trend_alignment = 1
        elif action == 2 and current_market_trend == 1:  # Close short in bullish market
            trend_alignment = 1
        elif action != 0:
            trend_alignment = -1
        return trend_alignment
    
    
    def calculate_risk_penalty(self, current_close_price):
        risk_penalty = 0.0
        if self.open_orders:
            last_order = self.open_orders[-1]
            stop_loss = 1.05 * last_order['entry_price']
            if current_close_price >= stop_loss:
                risk_penalty = 0.02 * last_order['amount'] * current_close_price
        return risk_penalty
    
    
    def calculate_take_profit(self, current_close_price):
        take_profit_reward = 0.0
        if self.open_orders:
            last_order = self.open_orders[-1]
            take_profit = 0.95 * last_order['entry_price']  # 5% below the entry price for a short
            if current_close_price <= take_profit:
                take_profit_reward = 0.02 * last_order['amount'] * current_close_price  # Example reward calculation
        return take_profit_reward
    
    
    def calculate_time_penalty(self):
        return self.current_step / self.max_steps
    
    
    
    def total_asset_in_zar(self):
        current_close_price = float(self.data.iloc[self.current_step]['closing_price'])
        btc_in_zar = self.btc_balance * current_close_price
        return self.zar_balance + btc_in_zar
    
    
    def total_asset_in_btc(self):
        current_close_price = float(self.data.iloc[self.current_step]['closing_price'])
        zar_in_btc = self.zar_balance / current_close_price
        return self.btc_balance + zar_in_btc
    
    
    def calculate_min_max_closing_prices(self):
        return self.data['closing_price'].min(), self.data['closing_price'].max()
    
    
    def log_metrics(self, episode, total_reward):
        print(f"Episode: {episode}, Total Reward: {total_reward}")
        print(f"Total Asset in ZAR: {self.total_asset_in_zar()}")
        print(f"Total Asset in BTC: {self.total_asset_in_btc()}")

    def render(self, mode='human', episode=None):
        if mode == 'human':
            print(f"Episode: {episode}, Step: {self.current_step}, ZAR Balance: {self.zar_balance}, BTC Balance: {self.btc_balance}")
            print(f"Total Asset in ZAR: {self.total_asset_in_zar()}")
            print(f"Total Asset in BTC: {self.total_asset_in_btc()}")






