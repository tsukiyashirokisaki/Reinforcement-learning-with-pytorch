from maze_env import Maze
from RL_brain import DeepQNetwork
import matplotlib.pyplot as plt
import numpy as np

def run_maze():
    step = 0
    rewards = []
    for episode in range(300):
        # initial observation
        observation = env.reset()
        r = 0
        iteration = 0
        while True:
            # fresh env
            iteration +=1
            env.render()

            # RL choose action based on observation
            action = RL.choose_action(observation)

            # RL take action and get next observation and reward
            observation_, reward, done = env.step(action)
            r += reward
            RL.store_transition(observation, action, reward, observation_)

            if (step > 200) and (step % 5 == 0):
                RL.learn()

            # swap observation
            observation = observation_

            # break while loop when end of this episode
            if done:
                r/= iteration
                print("epoch= %d reward= %.3f"%(episode+1,r))
                rewards.append(r)
                break
            step += 1

    # end of game
    plt.plot(np.arange(300),rewards)
    plt.ylabel('moving average reward')
    plt.xlabel('training steps')
    plt.show()
    print('game over')
    env.destroy()
    

if __name__ == "__main__":
    # maze game
    env = Maze()
    RL = DeepQNetwork(env.n_actions, env.n_features,
                      learning_rate=0.01,
                      reward_decay=0.9,
                      e_greedy=0.9,
                      replace_target_iter=200,
                      memory_size=2000,
                      # output_graph=True
                      )
    env.after(100, run_maze)
    env.mainloop()
    RL.plot_cost()