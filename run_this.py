import random

import matplotlib.pyplot as plt
import numpy as np

from DQN import DeepQNetwork
from env import Env
from memory import Memory
from sklearn.preprocessing import MinMaxScaler, StandardScaler

N_VM = 7
EPISODES = 250
MINI_BATCH = 128
MEMORY_SIZE = 10000
N_AGENT = 2


def run_env():
    step = 0
    for episode in range(EPISODES):
        rwd = [0.0, 0.0]
        obs = env.reset()
        # print(episode)

        while True:
            step += 1

            q_value = []
            if np.random.uniform() < dqn[0].epsilon:
                for i in range(N_AGENT):
                    # TODO(hang): standardized q_value
                    # normalize = scaler.fit_transform(dqn[i].choose_action(obs[i]).reshape(-1, 1)).reshape(7)
                    # q_value.append(normalize)
                    q_value.append(dqn[i].choose_action(obs[i]))
                q_sum = []
                for i in range(len(q_value[0])):
                    q_sum.append(q_value[0][i] + q_value[1][i])
                action = np.argmax(q_sum)
                # print(env.task, action)
            else:
                action = np.random.randint(0, env.n_actions - 1)

            obs_, reward, done = env.step(action)
            # print(obs_)
            rwd[0] += reward[0]
            rwd[1] += reward[1]

            # TODO(hang): each agent not only observes his own state but also others state
            for i in range(N_AGENT):
                memories[i].remember(obs[i], action, reward[i], obs_[i], done[i])
                size = memories[i].pointer
                batch = random.sample(range(size), size) if size < MINI_BATCH else random.sample(
                    range(size), MINI_BATCH)
                if step > 200 and step % 5 == 0:
                    dqn[i].learn(*memories[i].sample(batch))

            obs = obs_

            if done[0]:
                if episode == EPISODES - 1:
                    print(env.task_exec)
                    print(max(env.vm_time))
                    print(np.sum(env.vm_cost))
                    print(env.vm_time)
                    print(env.vm_cost)
                if episode % 10 == 0:
                    print(
                        'episode:' + str(episode) + ' steps:' + str(step) +
                        ' reward0:' + str(rwd[0]) + ' reward1:' + str(rwd[1]) +
                        ' eps_greedy0:' + str(dqn[0].epsilon) + ' eps_greedy1:' + str(dqn[1].epsilon))

                for i in range(N_AGENT):
                    rewards[i].append(rwd[i])
                break


if __name__ == '__main__':
    rewards = [[], []]

    scaler = StandardScaler()

    env = Env(N_VM, N_AGENT)

    memories = [Memory(MEMORY_SIZE) for i in range(N_AGENT)]

    dqn = [DeepQNetwork(env.n_actions, env.n_features, i,
                        learning_rate=0.001,
                        replace_target_iter=200,
                        e_greedy_increment=5e-5
                        ) for i in range(N_AGENT)]

    run_env()

    # for i in range(N_AGENT):
    #     dqn[i].plot_cost()

    plt.figure(1)
    plt.plot(np.arange(len(rewards[0])), rewards[0])
    plt.plot(np.arange(len(rewards[0])), [139 for i in range(len(rewards[0]))])
    plt.ylabel('reward 0')
    plt.xlabel('episode')
    # plt.show()

    plt.figure(2)
    plt.plot(np.arange(len(rewards[1])), rewards[1])
    plt.plot(np.arange(len(rewards[1])), [139 for i in range(len(rewards[0]))])
    plt.ylabel('reward 1')
    plt.xlabel('episode')
    plt.show()
