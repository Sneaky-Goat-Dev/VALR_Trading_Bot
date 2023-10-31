import argparse
import optuna
from trading_env_revised import TradingEnv
from stable_baselines3 import PPO
import json
import matplotlib.pyplot as plt

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

def evaluate_model(model, env):
    total_rewards = 0
    total_steps = 1000
    rewards = []
    obs = env.reset()
    
    for i in range(total_steps):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        total_rewards += reward
        rewards.append(reward)
        print(f'Total Reward: {total_rewards}')
        print(f'Reward: {reward}')
        env.render()
        if done:
            obs = env.reset()
    
    plot_rewards(rewards, total_steps)
    
    return total_rewards


def plot_rewards(rewards, total_steps):
    plt.figure(figsize=(10, 5))
    plt.plot(range(total_steps), rewards, label='Rewards per Step')
    plt.xlabel('Step')
    plt.ylabel('Reward')
    plt.title('Rewards over Time')
    plt.legend()
    plt.grid(True)
    plt.show()

def objective(trial):
    # Hyperparameters for the trading environment
    time_penalty = trial.suggest_categorical('time_penalty', [-0.01, -0.05, -0.1, -0.5])
    max_steps = trial.suggest_int('max_steps', 10, 100)
    time_threshold = trial.suggest_int('time_threshold', 5, 20)

    # Hyperparameters for the PPO model
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-2)
    batch_size = trial.suggest_categorical('batch_size', [8, 16, 32, 64, 128])
    n_steps = trial.suggest_int('n_steps', 128, 2048)
    gamma = trial.suggest_uniform('gamma', 0.9, 0.9999)
    gae_lambda = trial.suggest_uniform('gae_lambda', 0.8, 1.0)
    
    env = TradingEnv(time_penalty=time_penalty, max_steps=max_steps, time_threshold=time_threshold)
    
    model = PPO("MlpPolicy", env, verbose=1, 
                learning_rate=learning_rate, 
                batch_size=batch_size,
                n_steps=n_steps,
                gamma=gamma,
                gae_lambda=gae_lambda)
    
    model.learn(total_timesteps=20000)
    return evaluate_model(model, env)



def main(mode):
    if mode == "train":
        best_reward = float('-inf')  # Initialize with negative infinity

        def callback(study, trial):
            nonlocal best_reward
            if study.best_value > best_reward:
                env = TradingEnv(time_penalty=study.best_params['time_penalty'],
                                max_steps=study.best_params['max_steps'],
                                time_threshold=study.best_params['time_threshold'])

                best_model = train_model(env, 
                                        study.best_params['learning_rate'], 
                                        study.best_params['batch_size'],
                                        study.best_params['n_steps'],
                                        study.best_params['gamma'],
                                        study.best_params['gae_lambda'])

                total_rewards = evaluate_model(best_model, env)
                best_reward = save_best_model(best_model, study.best_params, total_rewards, best_reward)

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=100, callbacks=[callback])

        print('Number of finished trials:', len(study.trials))
        print('Best trial:')
        trial = study.best_trial

        print(f"Value: {trial.value}")
        print("Params:")
        for key, value in trial.params.items():
            print(f"{key}: {value}")

    elif mode == "test":
        print("Loading and Evaluating the best model...")
        params = load_best_params()
        env = TradingEnv(time_penalty=params['time_penalty'],
                         max_steps=params['max_steps'],
                         time_threshold=params['time_threshold'])
        best_model = PPO.load("best_trader")  # Make sure the path is consistent
        total_rewards = evaluate_model(best_model, env)
        print(f"Total rewards from the best model: {total_rewards}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train or Evaluate a trading model.')
    parser.add_argument('mode', type=str, choices=["train", "test"],
                        help='Mode to run the script in. Options are "train" or "test"')
    args = parser.parse_args()
    main(args.mode)