import argparse
import optuna
from optuna.pruners import MedianPruner
from trading_env_btc import TradingEnv
from stable_baselines3 import PPO
import json
import matplotlib.pyplot as plt
import matplotlib.animation as animation
# plt.ion()
from joblib import parallel_backend

def plot_rewards(rewards, total_steps):
    plt.figure(figsize=(10, 5))
    plt.plot(range(total_steps), rewards, label='Rewards per Step')
    plt.xlabel('Step')
    plt.ylabel('Reward')
    plt.title('Rewards over Time')
    plt.legend()
    plt.grid(True)
    plt.show()

def save_best_model(model, params, total_rewards, best_rewards, model_path="best_trader", params_path="best_params.json"):
    if total_rewards > best_rewards:
        model.save(model_path)
        with open(params_path, "w") as f:
            json.dump(params, f)
        return total_rewards
    return best_rewards

def train_model(env, learning_rate, batch_size, n_steps, gamma, gae_lambda):
    model = PPO("MlpPolicy", env, verbose=1, 
                learning_rate=learning_rate, 
                batch_size=batch_size,
                n_steps=n_steps,
                gamma=gamma,
                gae_lambda=gae_lambda)
    model.learn(total_timesteps=20000)
    return model

def load_best_params(params_path="best_params.json"):
    with open(params_path, "r") as f:
        params = json.load(f)
    return params


# return for testing
# Simple evaluate function that prints out the rewards
def test_evaluate_model(model, env):
    total_rewards = 0
    total_steps = 1000
    obs = env.reset()
    for i in range(total_steps):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        env.render()
        total_rewards += reward
        if(done):
            print(f'Total Rewards for the Best Model: {total_rewards}')
    return total_rewards


def evaluate_model(model, env):
    total_rewards = 0
    total_steps = 1000
    obs = env.reset()
    
    min_price, max_price = env.calculate_min_max_closing_prices()
    print(f"Min Closing Price: {min_price}, Max Closing Price: {max_price}")

    # Set up the figure and axis
    fig, ax = plt.subplots()
    ax.set_xlim(0, total_steps)  # Update the x-axis range to match the total number of 
    ax.set_ylim([min_price, max_price])  # Set min and max price based on your data

    price_line, = ax.plot([], [], lw=2)  # Line for price over time
    # trade_marker, = ax.plot([], [], 'ro')  # Markers for trades
    green_dots, = ax.plot([], [], 'go', markersize=2)  # Green dots for action type 1
    blue_dots, = ax.plot([], [], 'ro', markersize=2)   # Blue dots for action type 2
    

    def init():
        price_line.set_data([], [])
        green_dots.set_data([], [])
        blue_dots.set_data([], [])
        return price_line, green_dots, blue_dots

    def update(frame):
        nonlocal total_rewards
        nonlocal obs  
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        total_rewards += reward
        
        # Update your data lists for price and trades
        price_data = env.closing_prices[:frame]  
        trade_data_1 = [(trade['step'], trade['price']) for trade in env.trades if trade['step'] < frame and trade['action'] == 1]
        trade_data_2 = [(trade['step'], trade['price']) for trade in env.trades if trade['step'] < frame and trade['action'] == 2]
        
        # Update the line and marker data
        price_line.set_data(range(frame), price_data)
        
        if trade_data_1:
            green_dots.set_data(*zip(*trade_data_1))
        else:
            green_dots.set_data([], [])
        
        if trade_data_2:
            blue_dots.set_data(*zip(*trade_data_2))
        else:
            blue_dots.set_data([], [])
        
        return price_line, green_dots, blue_dots

    ani = animation.FuncAnimation(fig, update, frames=range(total_steps), init_func=init, blit=True)

    plt.show()

    return total_rewards

def objective(trial):
    # Hyperparameters for the trading environment
    time_penalty = trial.suggest_categorical('time_penalty', [-0.01, -0.05, -0.1, -0.5])
    max_steps = trial.suggest_int('max_steps', 10, 100)
    time_threshold = trial.suggest_int('time_threshold', 5, 20)


    # New hyperparameters to be tuned
    ALPHA = trial.suggest_float('ALPHA', 0.1, 1.0)
    BETA = trial.suggest_float('BETA', 0.01, 0.2)
    GAMMA = trial.suggest_float('GAMMA', 0.1, 1.0)
    DELTA = trial.suggest_float('DELTA', 0.1, 1.0)
    EPSILON = trial.suggest_float('EPSILON', 0.01, 0.1)
    ZETA = trial.suggest_float('ZETA', 0.1, 1.0)
    

    # Hyperparameters for the PPO model
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_categorical('batch_size', [8, 16, 32, 64, 128, 256, 512, 223])
    n_steps = trial.suggest_int('n_steps', 128, 2048)
    gamma = trial.suggest_float('gamma', 0.9, 0.9999)
    gae_lambda = trial.suggest_float('gae_lambda', 0.8, 1.0)
    
    env = TradingEnv(time_penalty=time_penalty, max_steps=max_steps, time_threshold=time_threshold,
                     ALPHA=ALPHA, BETA=BETA, GAMMA=GAMMA, DELTA=DELTA, EPSILON=EPSILON, ZETA=ZETA)
    
    model = train_model(env, learning_rate, batch_size, n_steps, gamma, gae_lambda)
    return evaluate_model(model, env)

def main(mode):
    if mode == "train":
        best_reward = float('-inf')

        def callback(study, trial):
            nonlocal best_reward
            if study.best_value > best_reward:
                env = TradingEnv(time_penalty=study.best_params['time_penalty'],
                                 max_steps=study.best_params['max_steps'],
                                 time_threshold=study.best_params['time_threshold'],
                                 ALPHA=study.best_params['ALPHA'],
                                 BETA=study.best_params['BETA'],
                                 GAMMA=study.best_params['GAMMA'],
                                 DELTA=study.best_params['DELTA'],
                                 EPSILON=study.best_params['EPSILON'],
                                 ZETA=study.best_params['ZETA'])

                best_model = train_model(env, 
                                         study.best_params['learning_rate'], 
                                         study.best_params['batch_size'],
                                         study.best_params['n_steps'],
                                         study.best_params['gamma'],
                                         study.best_params['gae_lambda'])

                total_rewards = evaluate_model(best_model, env)
                best_reward = save_best_model(best_model, study.best_params, total_rewards, best_reward)

        pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=30)
        study = optuna.create_study(direction='maximize', pruner=pruner)
        
        with parallel_backend('threading', n_jobs=4):
            study.optimize(objective, n_trials=1000, callbacks=[callback])

        print('Number of finished trials:', len(study.trials))
        print('Best trial:')
        trial = study.best_trial

        print(f"Value: {trial.value}")
        print("Params:")
        for key, value in trial.params.items():
            print(f"{key}: {value}")

        # Plotting feature importances
        optuna.visualization.plot_param_importances(study)

    elif mode == "test":
        print("Loading and Evaluating the best model...")
        params = load_best_params()
        env = TradingEnv(time_penalty=params['time_penalty'],
                         max_steps=params['max_steps'],
                         time_threshold=params['time_threshold'],
                         ALPHA=params['ALPHA'],
                         BETA=params['BETA'],
                         GAMMA=params['GAMMA'],
                         DELTA=params['DELTA'],
                         EPSILON=params['EPSILON'],
                         ZETA=params['ZETA'])
        best_model = PPO.load("best_trader")
        total_rewards = test_evaluate_model(best_model, env)
        print(f"Total rewards from the best model: {total_rewards}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train or Evaluate a trading model.')
    parser.add_argument('mode', type=str, choices=["train", "test"],
                        help='Mode to run the script in. Options are "train" or "test"')
    args = parser.parse_args()
    main(args.mode)




