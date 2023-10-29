from trading_env_revised import TradingEnv
from stable_baselines3 import PPO

def train(time_penalty):
    env = TradingEnv(time_penalty=time_penalty)
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=20000)
    model.save(f"ppo_trading_{time_penalty}")

def evaluate(time_penalty):
    env = TradingEnv(time_penalty=time_penalty)
    model = PPO.load(f"ppo_trading_{time_penalty}", env=env)

    total_rewards = 0
    obs = env.reset()
    for i in range(1000):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        env.render()
        total_rewards += reward
        if done:
            obs = env.reset()

    return total_rewards

if __name__ == '__main__':
    # List of time_penalty values to try
    time_penalties = [-0.01, -0.05, -0.1, -0.5]

    # Dictionary to store results
    results = {}

    # Train models with different time penalties
    for time_penalty in time_penalties:
        train(time_penalty)

    # Evaluate trained models
    for time_penalty in time_penalties:
        total_rewards = evaluate(time_penalty)
        results[time_penalty] = total_rewards
        print(f"Time Penalty: {time_penalty}, Average Total Rewards: {total_rewards}")

    # Print and get the optimal time_penalty
    optimal_time_penalty = max(results, key=results.get)
    print(f"The optimal time penalty is {optimal_time_penalty} with average total rewards of {results[optimal_time_penalty]}")

    # Evaluate the best model
    print("Evaluating the best model...")
    best_model_rewards = evaluate(optimal_time_penalty)
    print(f"Total rewards from the best model with time penalty {optimal_time_penalty}: {best_model_rewards}")