import numpy as np
import math
import abc


# Rounding only for floats like 1.0, 3.0, ...
def my_round(x):
    return math.floor(x + 1e-2)


class GridWorld():
    def __init__(self, length_X=10, length_Y=10, gamma=1.0):
        self.length_X = my_round(length_X)
        self.length_Y = my_round(length_Y)
        self.state_size = length_X * length_Y
        # Discount rate
        self.gamma = gamma
        # 0 - Up, 1 - Down, 2 - Left, 3 - Right
        self.actions = np.arange(4)
        self.states = np.arange(self.state_size)
        # Initialize transition matrix using a simplest one, i.e. deterministic transition
        self.transitions = np.zeros((4, self.state_size, self.state_size), float)
        self.init_deterministic_transitions()
        # Terminal state (for episodic setting), using the first one and the last one to initialize
        self.starting_state = 0
        self.terminal_states = [self.state_size - 1]

    @abc.abstractmethod
    def state_transition(self, state, action):
        pass

    # Initialize a deterministic transition matrix, we can also define other transition matrix
    def init_deterministic_transitions(self):
        for direction in range(4):
            for state in range(self.state_size):
                next_state = self.get_next_number_from_direction(direction, state)
                self.transitions[direction][state][next_state] = 1.0
        return

    # Print numbers (e.g. reward or Q-value) at each position
    def print_on_state_map(self, A, silent=False):
        assert A.shape == (self.state_size,)
        # matrix_A is the reflection (in some sense) of state map, we want to print matrix_A directly
        matrix_A = np.zeros((self.length_Y, self.length_X), float)
        for state in range(self.state_size):
            x, y = self.number_to_coordinate(state)
            matrix_A[self.length_Y-y-1][x] = A[state]
        if not silent:
            print(matrix_A)
        return matrix_A

    def coordinate_to_number(self, x, y):
        return my_round(y * self.length_X + x)

    def number_to_coordinate(self, state):
        y = np.floor(state / self.length_X)
        x = state - y * self.length_X
        return my_round(x), my_round(y)

    # From the current number and direction (not action) to next number
    def get_next_number_from_direction(self, direction, state):
        x, y = self.number_to_coordinate(state)
        if direction == 0:
            if y == self.length_Y - 1:
                next_state = state
            else:
                next_state = self.coordinate_to_number(x, y+1)
        elif direction == 1:
            if y == 0:
                next_state = state
            else:
                next_state = self.coordinate_to_number(x, y-1)
        elif direction == 2:
            if x == 0:
                next_state = state
            else:
                next_state = self.coordinate_to_number(x-1, y)
        elif direction == 3:
            if x == self.length_X - 1:
                next_state = state
            else:
                next_state = self.coordinate_to_number(x+1, y)
        else:
            raise Exception('Direction value error, direction must be one of {0, 1, 2, 3}.')
        return next_state


class SimpleGridWorld(GridWorld):
    def __init__(self, length_X=10, length_Y=10, gamma=1.0):
        GridWorld.__init__(self, length_X, length_Y, gamma)
        # For simplicity we consider the reward only depends on the current state, thus it is just a vector
        self.state_reward = -np.ones((self.state_size, 1))

    # Reward here only depends on the current state
    def state_transition(self, state, action):
        next_state = np.random.choice(a=self.states, size=1, p=self.transitions[action][state])[0]
        reward = self.state_reward[next_state]
        return next_state, reward


class WindyCliffGridWorld(SimpleGridWorld):
    # See readme for more details, we have to reassign the transition matrix, reward and terminal state
    def __init__(self, length_X=10, length_Y=10, gamma=1.0, theta=1.0):
        SimpleGridWorld.__init__(self, length_X, length_Y, gamma)
        self.punish = -100  # Punish for falling off a cliff
        self.award = 100
        self.terminal_states = range(1, self.length_X)
        self.theta = theta
        # Modify reward
        for state in range(1, self.length_X-1):
            self.state_reward[state] = self.punish
        self.state_reward[self.length_X - 1] = self.award
        # Modify transition matrix
        self.transitions = np.zeros((4, self.state_size, self.state_size), float)
        for action in range(4):
            for state in range(self.state_size):
                # Cliff
                if state in range(1, self.length_X):
                    self.transitions[action][state][0] = 1.0
                # Not cliff
                else:
                    # Down
                    if action == 1:
                        next_state = self.get_next_number_from_direction(1, state)
                        self.transitions[1][state][next_state] = 1.0
                    # Not down
                    else:
                        intend_next_state = self.get_next_number_from_direction(action, state)
                        accident_next_state = self.get_next_number_from_direction(1, state)
                        self.transitions[action][state][intend_next_state] += 1 - self.theta / 3
                        self.transitions[action][state][accident_next_state] += self.theta / 3


if __name__ == '__main__':
    env = WindyCliffGridWorld(6, 3)
    env.print_on_state_map(np.arange(18))
