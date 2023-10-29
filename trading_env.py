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
        self.actions_taken = []
        self.order_number = 0  # Initialize order number
        self.actions_with_order_numbers = []  # Store actions with their order numbers

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
        self.closed_orders = []

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
    
    def soft_reset(self):
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
        if action == 10:  # Buy to open a 'Long'
            if self.zar_balance > 0 and len(self.open_orders) < self.max_open_orders:
                max_risk_zar = self.zar_balance * risk_percent
                buy_amount = max_risk_zar / closing_price

                self.zar_balance -= max_risk_zar
                self.total_asset_in_trade += max_risk_zar

                self.open_orders.append({
                    'type': 'buy',
                    'trade_type': 'long',  # Specify the trade as a 'long'
                    'amount': buy_amount,
                    'price': closing_price,
                    'time_open': 0,
                    'order_number': self.order_number,
                    'matching_order_number': None
                })

        elif action == 11:  # Sell to open a 'Short'
            if self.btc_balance > 0 and len(self.open_orders) < self.max_open_orders:
                max_risk_btc = self.btc_balance * risk_percent
                self.btc_balance -= max_risk_btc
                self.total_asset_in_trade += (max_risk_btc * closing_price)

                self.open_orders.append({
                    'type': 'sell',
                    'trade_type': 'short',  # Specify the trade as a 'short'
                    'amount': max_risk_btc,
                    'price': closing_price,
                    'time_open': 0,
                    'order_number': self.order_number,
                    'matching_order_number': None
                })

        if 0 <= action < 5:  # Close a 'Long'
            order_idx_to_close = action
            if order_idx_to_close < len(self.open_orders):
                order = self.open_orders.pop(order_idx_to_close)
                if order['trade_type'] == 'long':  # Check if it's a 'long' trade
                    self.btc_balance += order['amount']
                    self.total_asset_in_trade -= (order['amount'] * order['price'])
                    trade_profit_loss = (closing_price - order['price']) * order['amount']
                    order['matching_order_number'] = self.order_number
                    self.closed_orders.append(order)

        elif 5 <= action < 10:  # Close a 'Short'
            order_idx_to_close = action - 5
            if order_idx_to_close < len(self.open_orders):
                order = self.open_orders.pop(order_idx_to_close)
                if order['trade_type'] == 'short':  # Check if it's a 'short' trade
                    self.zar_balance += (order['amount'] * order['price'])
                    self.total_asset_in_trade -= (order['amount'] * order['price'])
                    trade_profit_loss = (order['price'] - closing_price) * order['amount']
                    order['matching_order_number'] = self.order_number
                    self.closed_orders.append(order)

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


    def render(self, mode='human', episode=None):
        if len(self.actions_with_order_numbers) == 0 or self.current_step == 0:
            print("Not enough data to render. Continue the episode.")
            return

        window_data = self.data.iloc[:self.current_step]
        actions_window = self.actions_with_order_numbers[:self.current_step]

        plt.figure(figsize=(10, 6))

        # Plot the closing prices
        plt.plot(range(len(window_data)), window_data['closing_price'], marker='o')

        # Annotate the buy/sell/hold actions
        for i, (index, row) in enumerate(window_data.iterrows()):
            action, order_number = actions_window[i]
            trade_type = None
            matching_order_number = None
            
            if action in [10, 11]:
                for order in self.open_orders:
                    if order['order_number'] == order_number:
                        trade_type = order['trade_type']
                        matching_order_number = order.get('matching_order_number')
                        break

                label = f"{trade_type.capitalize()} {order_number} (Match: {matching_order_number})"
                plt.annotate(label,
                            (i, row['closing_price']),
                            textcoords="offset points",
                            xytext=(0, 10),
                            ha='center',
                            color='g' if action == 10 else 'r')

        if episode is not None:
            plt.title(f'Episode {episode}: Trading actions from start to step {self.current_step}')
        else:
            plt.title(f'Trading actions from start to step {self.current_step}')

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




best_reward = -np.inf  # Initialize best_reward to negative infinity
best_model = None  # Initialize best_model to None

def train():
    # Main script

    def objective(trial):
        try:
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
                print(f'UNREALISED BALANCE: {env.zar_balance + env.total_asset_in_trade}')
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
        except Exception as e:
            print(f"An exception occurred: {e}")
            return -1e9  # Return a large negative value

    # Create a study object and specify the direction is maximize.
    study = optuna.create_study(direction='maximize')

    # Optimize the study, the objective function is passed in as the first argument.
    study.optimize(objective, n_trials=5)  # You can adjust the number of trials

    # Results
    print('Number of finished trials: ', len(study.trials))
    print('Best trial:')
    trial = study.best_trial

    print('Value: ', trial.value)

    print('Params: ')
    for key, value in trial.params.items():
        print(f'    {key}: {value}')




def plot_best():
    # Load the best model
    best_model = PPO.load("best_model")

    # Create the environment
    env = TradingEnv()

    # Run multiple episodes with the best model
    for episode in range(1, 100):  # You can set the number of episodes you want, starting from 1
        obs = env.soft_reset()  # Use soft_reset to keep actions and steps from previous episodes
        for _ in range(2):  # or however many steps you want to run per episode
            action, _ = best_model.predict(obs)
            obs, reward, done, _ = env.step(action)
            if done:
                break
        # Render the results after each episode
        if episode == 99:
            env.render(episode=episode)
            

    # Reset actions and steps, as you've plotted everything
    env.actions_with_order_numbers = []
    env.current_step = 0



# train()
plot_best()