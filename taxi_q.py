import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pickle

def run(episodes, is_training=True, render=False):
    try:
        # Try to create the environment with rendering
        env = gym.make('Taxi-v3', 
                       render_mode='human' if render else None)
    except Exception as e:
        # Fallback to non-rendering mode if there's an issue
        print(f"Rendering failed: {e}")
        env = gym.make('Taxi-v3')
        render = False

    if(is_training):
        q = np.zeros((env.observation_space.n, env.action_space.n))
    else:
        with open('taxi.pkl', 'rb') as f:
            q = pickle.load(f)

    learning_rate_a = 0.9
    discount_factor_g = 0.9
    epsilon = 1
    epsilon_decay_rate = 0.0001
    rng = np.random.default_rng()

    rewards_per_episode = np.zeros(episodes)

    for i in range(episodes):
        state, _ = env.reset()
        terminated = False
        truncated = False

        rewards = 0
        while(not terminated and not truncated):
            if is_training and rng.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q[state,:])

            new_state, reward, terminated, truncated, _ = env.step(action)

            rewards += reward

            if is_training:
                q[state,action] = q[state,action] + learning_rate_a * (
                    reward + discount_factor_g * np.max(q[new_state,:]) - q[state,action]
                )

            state = new_state

        epsilon = max(epsilon - epsilon_decay_rate, 0)

        if(epsilon==0):
            learning_rate_a = 0.0001

        rewards_per_episode[i] = rewards

    env.close()

    sum_rewards = np.zeros(episodes)
    for t in range(episodes):
        sum_rewards[t] = np.sum(rewards_per_episode[max(0, t-100):(t+1)])
    
    plt.figure(figsize=(10,6))
    plt.plot(sum_rewards)
    plt.title('Rewards over Episodes')
    plt.xlabel('Episodes')
    plt.ylabel('Cumulative Rewards')
    plt.savefig('taxi.png')
    plt.close()

    if is_training:
        with open("taxi.pkl","wb") as f:
            pickle.dump(q, f)

if __name__ == '__main__':
    # Training phase
    run(15000)
    
    # Rendering phase (with error handling)
    try:
        run(10, is_training=False, render=True)
    except Exception as e:
        print(f"Could not render environment: {e}")
        print("Running without rendering...")
        run(10, is_training=False, render=False)