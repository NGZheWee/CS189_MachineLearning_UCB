"""
Author: Aryan Jain
"""

import random
from random import shuffle, randrange
import numpy as np
import matplotlib.pyplot as plt

class Env:
    def __init__(self, rows: int, columns: int, openness: float, epsilon: float = 0.0):
        self._rows = rows
        self._columns = columns
        self._epsilon = epsilon
        self._openness = openness
        self._nodes : dict[tuple[int, int], Node] = {}

        self.create_env()

    def init_env(self):
        """
        Call init_env before collecting any trajectories / stepping through the environment.
        """
        self._hidden_state = None
        self._states_so_far = []

    def create_env(self) -> None:
        """
        This helper function creates the actual environment that the agent navigates in.
        Feel free to read this code if you want; however, DO NOT CHANGE THIS IN ANY WAY
        Feel free to also skip it; you can complete this assignment without understanding
        how exactly the environment was generated.

        Maze generation code taken from:
        https://rosettacode.org/wiki/Maze_generation
        """
        vis = [[0] * self._columns + [1] for _ in range(self._rows)] + [[1] * (self._columns + 1)]
        ver = [["| "] * self._columns + ['|'] for _ in range(self._rows)] + [[]]
        hor = [["+-"] * self._columns + ['+'] for _ in range(self._rows + 1)]

        def walk(x, y):
            vis[y][x] = 1

            d = [(x - 1, y), (x, y + 1), (x + 1, y), (x, y - 1)]
            shuffle(d)
            for (xx, yy) in d:
                if vis[yy][xx]: continue
                if xx == x: hor[max(y, yy)][x] = "+ "
                if yy == y: ver[y][max(x, xx)] = "  "
                walk(xx, yy)

        walk(randrange(self._columns), randrange(self._rows))

        rows = []
        for i, (a, b) in enumerate(zip(hor, ver)):
            if 0 < i < self._rows:
                for j in range(1, len(a) - 1):
                    if a[j] == "+-" and random.random() < self._openness:
                        a[j] = "+ "
            if 0 <= i < self._rows:
                for j in range(1, len(b) - 1):
                    if b[j] == "| " and random.random() < self._openness:
                        b[j] = "  "
            rows.append(list(''.join(a)))
            rows.append(list(''.join(b)))
        rows.pop()
        self.maze = rows

        for i, idx1 in enumerate(range(1, len(self.maze), 2)):
            for j, idx2 in enumerate(range(1, len(self.maze[idx1]), 2)):
                node = Node(i, j)
                if self.maze[idx1][idx2 - 1] == "|":
                    node.blocked["l"] = 1
                if self.maze[idx1][idx2 + 1] == "|":
                    node.blocked["r"] = 1
                if self.maze[idx1 - 1][idx2] == "-":
                    node.blocked["u"] = 1
                if self.maze[idx1 + 1][idx2] == "-":
                    node.blocked["d"] = 1
                self._nodes[(i, j)] = node

                if i > 0:
                    self._nodes[(i - 1, j)].neighbors.append(node)
                    node.neighbors.append(self._nodes[(i - 1, j)])
                if j > 0:
                    self._nodes[(i, j - 1)].neighbors.append(node)
                    node.neighbors.append(self._nodes[(i, j - 1)])

    def set_epsilon(self, epsilon: float) -> None:
        self._epsilon = epsilon

    def get_neighbors(self, i: int, j: int) -> list[tuple[int, int]]:
        """
        Returns all the states that are accessible from state (i, j).
        Equivalently, these are the states that are not blocked from state (i, j) by a wall.
        You may find this function useful when computing the emission probabilities.
        """
        node = self._nodes[(i, j)]
        moves = [move for move, blocked in node.blocked.items() if not blocked]
        next_states = []
        if "r" in moves:
            next_states.append((i, j + 1))
        if "l" in moves:
            next_states.append((i, j - 1))
        if "u" in moves:
            next_states.append((i - 1, j))
        if "d" in moves:
            next_states.append((i + 1, j))
        return next_states

    def step(self) -> np.ndarray:
        """
        This function performs a random walk.
        Every time step() is called, the agent moves to one of its neighbors and emits a
        sensor observation (that is then corrupted using the probability of error epsilon).
        The true hidden state of the agent, which the environment tracks internally, is
        also updated.
        """

        if self._hidden_state is None:
            self._hidden_state = (random.randrange(self._rows), random.randrange(self._columns))
        else:
            # Find the node associated with the current robot hidden state
            node = self._nodes[self._hidden_state]
            # Retrieve the set of states that the robot can transition to
            # and choose one uniformly at random
            # This simulates the underlying hidden markov chain
            moves = [move for move, blocked in node.blocked.items() if not blocked]
            move = random.choice(moves)
            if move == "l":
                self._hidden_state = (self._hidden_state[0], self._hidden_state[1] - 1)
            if move == "r":
                self._hidden_state = (self._hidden_state[0], self._hidden_state[1] + 1)
            if move == "u":
                self._hidden_state = (self._hidden_state[0] - 1, self._hidden_state[1])
            if move == "d":
                self._hidden_state = (self._hidden_state[0] + 1, self._hidden_state[1])
        # Once you have arrived at the new state, emit an observation
        # First find the true observation
        true_reading = self._nodes[self._hidden_state]._true_sensor_reading()
        # Then, corrupt it to simulate faulty sensors
        observed_reading = []
        observed_reading.append(1 - true_reading[0] if random.random() < self._epsilon else true_reading[0])
        observed_reading.append(1 - true_reading[1] if random.random() < self._epsilon else true_reading[1])
        observed_reading.append(1 - true_reading[2] if random.random() < self._epsilon else true_reading[2])
        observed_reading.append(1 - true_reading[3] if random.random() < self._epsilon else true_reading[3])
        self._states_so_far.append(self._hidden_state)
        return np.array(observed_reading)

    def compute_accuracy(self, predictions: np.ndarray) -> np.ndarray:
        """
        Returns the cumulative accuracy array for the given set of predictions.
        The ith index will contain the accuracy of the first i predictions when compared against the first i true hidden states.
        """
        return np.cumsum(np.all(predictions == np.array(self._states_so_far), axis=-1)) / np.arange(1, len(self._states_so_far) + 1) * 100

    def plot_env(self) -> None:
        """
        Saves the environment layout to env_layout.png.
        """
        _, ax = plt.subplots()
        markersize = 240 / self._rows + 1
        ax.set_aspect("equal")
        # Plot the maze
        for i in range(len(self.maze)):
            for j in range(len(self.maze[0])):
                if self.maze[i][j] == '|':
                    ax.plot(j, -i, 'k|', markersize=markersize)  # vertical wall
                elif self.maze[i][j] == '-':
                    ax.plot(j, -i, 'k_', markersize=markersize)  # horizontal wall
                elif self.maze[i][j] == ' ':
                    ax.plot(j, -i, 'w_', markersize=markersize)  # empty space

        ax.xaxis.set_tick_params(labelbottom=False)
        ax.yaxis.set_tick_params(labelleft=False)

        ax.set_xticks([])
        ax.set_yticks([])
        plt.savefig("./env_layout.png")

class Node:
    def __init__(self, i: int, j: int) -> None:
        self.i = i
        self.j = j
        self.neighbors: list[Node] = []
        self.blocked : dict[str, bool] = {"r" : 0, "l" : 0, "u" : 0, "d" : 0}

    def _true_sensor_reading(self) -> tuple[int]:
        return [int(self.blocked["l"]), int(self.blocked["r"]), int(self.blocked["u"]), int(self.blocked["d"])]
