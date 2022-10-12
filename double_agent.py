import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt


class Agent():

    def __init__(self, seed=1234, init_lower=10, init_upper=20, alpha=.3, epsilon=.2):
        seed_ = seed
        np.random.seed(seed_)
        # Create value table corresponding to board
        self.board = np.random.uniform(init_lower, init_upper, size=(5, 10))
        np.set_printoptions(precision=2, suppress=True)
        print(self.board)
        # Start location
        self.start_point = [2, 3]
        # Capture points
        self.T1 = [3, 1]
        self.T2 = [0, 9]
        # Rewards
        self.step_reward = -1
        self.capture_reward = 20
        # params
        self.eps = epsilon
        self.alpha = alpha
        self.gamma = .5

        # class vars
        self.current_state = self.start_point
        self.current_action = None
        self.ep_buffer = []
        self.complete = False
        self.tf1 = True
        self.tf2 = True
        self.steps = 0

    def get_random_action(self):
        valid_actions = []
        s = self.current_state
        if s[1] - 1 >= 0:
            valid_actions.append("left")
        if s[1] + 1 <= 9:
            valid_actions.append("right")
        if s[0] - 1 >= 0:
            valid_actions.append("up")
        if s[0] + 1 <= 4:
            valid_actions.append("down")
        idx = np.random.randint(0, len(valid_actions))
        self.current_action = valid_actions[idx]

    def get_highest_value_action(self):
        action = None
        highest = -500
        s = self.current_state
        if s[1] - 1 >= 0 and self.board[s[0]][s[1]-1] > highest:
            action = "left"
            highest = self.board[s[0]][s[1]-1]
        if s[1] + 1 <= 9 and self.board[s[0]][s[1]+1] > highest:
            action = "right"
            highest = self.board[s[0]][s[1]+1]
        if s[0] - 1 >= 0 and self.board[s[0] - 1][s[1]] > highest:
            action = "up"
            highest = self.board[s[0] - 1][s[1]]
        if s[0] + 1 <= 4 and self.board[s[0] + 1][s[1]] > highest:
            action = "down"
            highest = self.board[s[0] + 1][s[1]]
        self.current_action = action

    def get_reward_local(self):
        if self.current_state == self.T1 and self.tf1 == True:
            self.tf1 = False
            return self.capture_reward
        if self.current_state == self.T2 and self.tf2 == True:
            self.tf2 = False
            return self.capture_reward
        return self.step_reward

    def get_next_state(self):
        a = self.current_action
        if a == "left":
            self.current_state[1] -= 1
        if a == "right":
            self.current_state[1] += 1
        if a == "up":
            self.current_state[0] -= 1
        if a == "down":
            self.current_state[0] += 1

    def get_next_value(self):
        highest = -500
        s = self.current_state
        if s[1] - 1 >= 0 and self.board[s[0]][s[1]-1] > highest:
            highest = self.board[s[0]][s[1]-1]
        if s[1] + 1 <= 9 and self.board[s[0]][s[1]+1] > highest:
            highest = self.board[s[0]][s[1]+1]
        if s[0] - 1 >= 0 and self.board[s[0] - 1][s[1]] > highest:
            highest = self.board[s[0] - 1][s[1]]
        if s[0] + 1 <= 4 and self.board[s[0] + 1][s[1]] > highest:
            highest = self.board[s[0] + 1][s[1]]
        return highest

    def take_action(self):
        # Has 1-eps chance of choosing the action with highest value
        s = self.current_state
        r = None
        if np.random.rand() < 1-self.eps:
            self.get_highest_value_action()
        else:
            self.get_random_action()
        current_val = self.board[self.current_state[0],
                                 self.current_state[1]]
        self.get_next_state()
        r = self.get_reward_local()
        if r == self.capture_reward:
            self.complete = True
        next_val = self.get_next_value()
        updated_val = current_val + self.alpha * \
            ((r + self.gamma * next_val) - current_val)
        self.board[self.current_state[0],
                   self.current_state[1]] = updated_val

    def take_step(self):
        self.take_action()
        if self.complete or self.steps > 100:
            return True
        self.steps += 1
        return self.complete

    def reset(self):
        self.current_state = self.start_point
        self.ep_buffer.clear()
        self.complete = False
        self.tf1 = True
        self.tf2 = True
        self.steps = 0


def run_episodes(num_episodes=30000):
    a1 = Agent()
    a2 = Agent(seed=1235)
    for i in range(num_episodes):
        complete1 = False
        complete2 = False
        flag = False
        while not flag:
            if not complete1:
                complete1 = a1.take_step()
                a2.tf1 = a1.tf1
                a2.tf2 = a1.tf2
            if not complete2:
                complete2 = a2.take_step()
                a1.tf1 = a2.tf1
                a1.tf2 = a2.tf2
            if complete2 == True and complete1 == True:
                flag = True
        a1.reset()
        a2.reset()

    fig, ax = plt.subplots()
    im = ax.imshow(a1.board)
    for i in range(5):
        for j in range(10):
            text = ax.text(j, i, float("{:.2f}".format(a1.board[i][j])),
                           ha="center", va="center", color="w")
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Values", rotation=-90, va="bottom")
    ax.set_title("Value Table Agent 1")
    fig.tight_layout()
    plt.show()
    fig, ax = plt.subplots()
    im = ax.imshow(a2.board)
    for i in range(5):
        for j in range(10):
            text = ax.text(j, i, float("{:.2f}".format(a2.board[i][j])),
                           ha="center", va="center", color="w")
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Values", rotation=-90, va="bottom")
    ax.set_title("Value Table Agent 2")
    fig.tight_layout()
    plt.show()

    print(a1.board)
    print(a2.board)


run_episodes()
