import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pickle

def run(episodes, is_training=True, render=False):

    env = gym.make('FrozenLake-v1', map_name="8x8", is_slippery=False, render_mode='human' if render else None)

    if (is_training): #if we are training, then initialize the Q table
        q = np.zeros((env.observation_space.n, env.action_space.n)) # create 64 x 4 array
    else: # otherwise load the model back into the q variable
        f = open('frozen_lake8x8.pk1', 'rb')
        q = pickle.load(f)
        f.close()

    learning_rate_a = 0.9 #alpha or learning rate (how quickly the algorithm is able to learn / determines to what extent the newly acquired information will override the old information.)
    discount_factor_g = 0.9 #gamma or discount factor (how the algorithm determines the value of future rewards)

    epsilon = 1 #1 = 100% random actions
    epsilon_decay_rate = 0.0001 # decrease randomness over time. also affects the min number of episodes needed to train. 1/0.0001 = 10,000
    rng = np.random.default_rng() #random number generator 

    rewards_per_episode = np.zeros(episodes) #track how training progresses

    for i in range(episodes):
        
        state = env.reset()[0] #sets state to 0, meaning the top left corner of the grid (63 is bottom right)
        terminated = False # true when fall in hole or reached goal
        truncated = False #true when actions > 200


        while (not terminated and not truncated):

            if is_training and rng.random() < epsilon: #if we are training and random number less than epsilon, take random action
                action = env.action_space.sample() # actions: 0=left, 1=down, 2=right, 3=up
            else: #otherwise follow Q table
                action = np.argmax(q[state,:])

            new_state, reward, terminated, truncated, _ = env.step(action) #execute action and return new states

            if is_training: #only update q table if we are training 
                q[state,action] = q[state,action] + learning_rate_a * (     #iteratively adjust the Q-values towards the expected long-term return following the Q-learning algorithm
                    reward + discount_factor_g * np.max(q[new_state,:]) - q[state,action] 
                )

            state = new_state

        epsilon = max(epsilon - epsilon_decay_rate, 0) #after each episode decrease epsilon until it reaches 0

        if (epsilon==0): #reduce learning rate: helps stabilize Q values after we are no longer exploring 
            learning_rate_a = 0.0001

        if reward == 1:
            rewards_per_episode[i] = 1

    env.close()

    sum_rewards = np.zeros(episodes)
    for t in range(episodes):
        sum_rewards[t] = np.sum(rewards_per_episode[max(0, t-100):(t+1)]) #show running sum of the rewards per 100 episodes
    plt.plot(sum_rewards)
    plt.savefig('frozen_lake8x8.png')
    
    if is_training: #only save q table if training 
        f = open("frozen_lake8x8.pk1", "wb")
        pickle.dump(q,f)
        f.close()

if __name__ == '__main__':
    #run(15000)
    
    run(1, is_training=False, render=True)