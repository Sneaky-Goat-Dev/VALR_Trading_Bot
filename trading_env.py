import gym
from gym import spaces
import mysql.connector
import numpy as np
import pandas as pd
from decouple import config
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
import optuna

class TradingEnv(gym.Env):
    
    def __init__(self):
        super(TradingEnv, self).__init__()

        columns_to_keep = [
            'opening_price', 'closing_price', 'high_price', 'low_price',
            'price_change', 'total_volume', 'volume_weighted_avg_price', 
            'number_of_trades', 'buy_volume', 'sell_volume',
            'simple_moving_average', 'exponential_moving_average',
            'rsi', 'macd', 'market_trend'
        ]
        
        self.max_steps = 10
        self.current_step = 0
        self.btc_balance = 0.0
        self.zar_balance = 2000.0
        self.initial_zar_balance = self.zar_balance
        self.total_profit = 0.0
        self.total_loss = 0.0
        self.total_asset_in_trade = 0.0
        self.time_penalty = -0.01
        self.time_threshold = 10
        self.high_buy_penalty = -0.02

        self.open_orders = []
        self.max_open_orders = 5
        
        self.action_space = spaces.Discrete(12)
        self.observation_space = spaces.Box(
            low=0, high=np.inf, shape=(self.max_steps, len(columns_to_keep)), dtype=np.float32
        )

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
        self.data = self.data[columns_to_keep]
        self.open_orders = []

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
        self.open_orders = []
        obs = self.data.iloc[self.current_step:self.current_step + self.max_steps].values.astype(np.float32)
        return obs
    
    def calculate_reward(self, trade_profit_loss, reward, open_orders):
        # Penalty for holding too long
        for order in open_orders:
            if 'time_open' in order and order['time_open'] > self.time_threshold:
                reward += self.time_penalty  # Notice that we're adding to the existing reward

        # Penalty or bonus for trade profit/loss
        if trade_profit_loss > 0:
            reward += trade_profit_loss  # Bonus reward for profitable trade
        elif trade_profit_loss < 0:
            reward += trade_profit_loss  # Penalty for a losing trade
            
        if self.total_profit >= 0:
            reward += self.total_profit - self.total_loss
            

        return reward


    def step(self, action):
        # print(f"Action taken: {action}")
        # print(f"Open orders: {self.open_orders}")
        
        self.current_step += 1
        reward = 0  # Initialize reward to 0
        trade_profit_loss = 0
        
        done = False
        if self.current_step >= self.max_steps:
            done = True
        
        current_state = self.data.iloc[self.current_step:self.current_step + self.max_steps]
        closing_price = float(current_state['closing_price'].iloc[-1])
        
        risk_percent = 0.01

        # Penalty for holding too long
        for order in self.open_orders:
            if 'time_open' not in order:
                order['time_open'] = 0
            else:
                order['time_open'] += 1
            if order['time_open'] > self.time_threshold:
                reward += self.time_penalty

        # Action logic
        if action == 10 and closing_price > current_state['simple_moving_average'].iloc[-1]:
            reward += self.high_buy_penalty

            if self.zar_balance > 0 and len(self.open_orders) < self.max_open_orders:
                max_risk_zar = self.zar_balance * risk_percent
                buy_amount = max_risk_zar / closing_price
                
                self.zar_balance -= max_risk_zar
                self.total_asset_in_trade += max_risk_zar
                
                self.open_orders.append({
                    'type': 'buy',
                    'amount': buy_amount,
                    'price': closing_price,
                    'time_open': 0  # Initialize time_open
                })

                # Penalty for buying too high
                if closing_price > current_state['simple_moving_average'].iloc[-1]:
                    reward += self.high_buy_penalty

        elif action == 11:
            if self.btc_balance > 0 and len(self.open_orders) < self.max_open_orders:
                max_risk_btc = self.btc_balance * risk_percent
                
                self.btc_balance -= max_risk_btc
                self.total_asset_in_trade += (max_risk_btc * closing_price)
                
                self.open_orders.append({
                    'type': 'sell',
                    'amount': max_risk_btc,
                    'price': closing_price,
                    'time_open': 0  # Initialize time_open
                })

        if 0 <= action < 5:
            order_idx_to_close = action
            if order_idx_to_close < len(self.open_orders):
                order = self.open_orders[order_idx_to_close]
                if order['type'] == 'buy':
                    self.btc_balance += order['amount']
                    self.total_asset_in_trade -= (order['amount'] * order['price'])
                    
                    trade_profit_loss = (closing_price - order['price']) * order['amount']
                    self.open_orders.pop(order_idx_to_close)

        elif 5 <= action < 10:
            order_idx_to_close = action - 5
            if order_idx_to_close < len(self.open_orders):
                order = self.open_orders[order_idx_to_close]
                if order['type'] == 'sell':
                    self.zar_balance += (order['amount'] * order['price'])
                    self.total_asset_in_trade -= (order['amount'] * order['price'])
                    
                    trade_profit_loss = (order['price'] - closing_price) * order['amount']
                    self.open_orders.pop(order_idx_to_close)
                    
        if trade_profit_loss > 0:
            self.total_profit += trade_profit_loss
        else:
            self.total_loss += trade_profit_loss

        reward = self.calculate_reward(trade_profit_loss, reward, self.open_orders)
        
        obs = self.data.iloc[self.current_step:self.current_step + self.max_steps].values.astype(np.float32)
        
        return obs, reward, done, {}
    
    
    def log_metrics(self, episode, total_reward):
        print(f"Episode: {episode}, Total Reward: {total_reward}")
        print(f"Total Asset in Trade: {self.total_asset_in_trade}")


    def render(self, mode='human'):
        if self.current_step < self.max_steps:
            print("Not enough data to render. Continue the episode.")
            return
        
        window_data = self.data.iloc[self.current_step - self.max_steps:self.current_step]
        actions_window = self.actions_taken[self.current_step - self.max_steps:self.current_step]  # Get the actions for the current window
        
        # Check if actions_window has enough elements
        if len(actions_window) < self.max_steps:
            print("Not enough actions to render. Continue the episode.")
            return
        
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

        plt.title(f'Trading actions over the last {self.max_steps} steps')
        plt.xlabel('Step')
        plt.ylabel('Closing Price')
        plt.grid()
        plt.show()





# env = TradingEnv()
# observation = env.reset()
# for _ in range(100):
#     action = env.action_space.sample()
#     print(f"Sampling action: {action}")
#     observation, reward, done, info = env.step(action)
#     if done:
#         print("Episode done, rendering...")
#         env.render()
#         observation = env.reset()


# Main script
best_reward = -np.inf  # Initialize best_reward to negative infinity
best_model = None  # Initialize best_model to None

def objective(trial):
    global best_reward, best_model  # Declare global variables

    # Hyperparameters to tune
    gamma = trial.suggest_float("gamma", 0.9, 0.99)
    n_steps = trial.suggest_int("n_steps", 16, 64)
    ent_coef = trial.suggest_float("ent_coef", 0.0, 0.1)
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
    
    # Create environment and model with new hyperparameters
    env = TradingEnv()
    model = PPO(
        "MlpPolicy", 
        env, 
        gamma=gamma, 
        n_steps=n_steps, 
        ent_coef=ent_coef, 
        learning_rate=learning_rate, 
        verbose=0  # Set to 0 to disable output during hyperparameter tuning
    )
    
    obs = env.reset()
    total_reward = 0
    for _ in range(10000):  # You can adjust the number of timesteps
        action, _ = model.predict(obs)
        obs, reward, done, _ = env.step(action)
        total_reward += reward
        
        print(f'TOTAL REWARD: {total_reward}')
        print(f'REWARD: {reward}')
        print(f'ZAR BALANCE: {env.zar_balance}')
        print(f'BTC BALANCE: {env.btc_balance}')
        print(f'ACTIVE TRADE BALANCE: {env.total_asset_in_trade}')
        print(f'TOTAL PROFIT: {env.total_profit}')
        print(f'TOTAL LOSS: {env.total_loss}')
        
        
        if done:
            obs = env.reset()

    # Check if this trial's total_reward is greater than best_reward
    if total_reward > best_reward:
        best_reward = total_reward
        best_model = model  # Save the best model
        model.save("best_model")  # Save the model to disk

    return total_reward  # Optuna aims to maximize this value

# Create a study object and specify the direction is maximize.
study = optuna.create_study(direction='maximize')

# Optimize the study, the objective function is passed in as the first argument.
study.optimize(objective, n_trials=100)  # You can adjust the number of trials

# Results
print('Number of finished trials: ', len(study.trials))
print('Best trial:')
trial = study.best_trial

print('Value: ', trial.value)

print('Params: ')
for key, value in trial.params.items():
    print(f'    {key}: {value}')
