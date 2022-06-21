import numpy as np
import pandas as pd

import utils


class GNN_env(object):
    def __init__(self, action_value):
        super(GNN_env, self).__init__()

        self.action_space = ['add', 'minus']
        self.action_value = action_value
        self.n_actions = len(self.action_space)

    def reward(self, current_acc, last_acc):

        if current_acc > last_acc:
            reward = 1
            done = True
        elif current_acc < last_acc:
            reward = -1
            done = True
        else:
            reward = 0
            done = True
        return reward, done

    def step(self, current_k, action):

        if action == 0:
            if current_k + self.action_value <= 1.0:
                new_k = current_k + self.action_value
            else:
                new_k = 1.0
        elif action == 1:
            if current_k - self.action_value >= -1.0:
                new_k = current_k - self.action_value
            else:
                new_k = -1.0

        return new_k

  
class QLearningTable:
    def __init__(self, actions, learning_rate=0.1, reward_decay=0.9, e_greedy=0.9):

        self.actions = actions
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy

        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)

    def choose_action(self, observation):
        self.check_state_exist(observation)

        if np.random.uniform() < self.epsilon:
            state_action = self.q_table.loc[observation, :]
            action = np.random.choice(state_action[state_action == np.max(state_action)].index)
        else:
            action = np.random.choice(self.actions)

        return action

    def learn(self, observation, action, reward, new_observation, done):
        self.check_state_exist(new_observation)

        q_predict = self.q_table.loc[observation, action]
        if not done:
            q_target = reward + self.gamma * self.q_table.loc[new_observation, :].max()
        else:
            q_target = reward
        self.q_table.loc[observation, action] += self.lr * (q_target - q_predict)

    def check_state_exist(self, state):
        if state not in self.q_table.index:
            self.q_table = self.q_table.append(pd.Series([0] * len(self.actions), index=self.q_table.columns, name=state))


def isTerminal(k_record, delta_k=0.05, last_epochs=20, start_epochs=150):

    if len(k_record) >= start_epochs:
        record_last = np.array(k_record[-last_epochs:])
        range_k = np.max(record_last) - np.min(record_last)

        if range_k <= delta_k:
            return True
        else:
            return False

    else:
        return False


def Run_QL(env, RL, current_acc, last_acc, last_k, current_k, action=None):

    if action is not None:
        reward, done = env.reward(current_acc, last_acc)
        RL.learn(str(last_k), action, reward, str(current_k), done)

    action = RL.choose_action(str(current_k))
    new_k = env.step(current_k, action)

    return current_k, new_k, action
