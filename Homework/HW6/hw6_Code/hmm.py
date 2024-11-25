"""
Author: Aryan Jain
"""
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend

import random
from env import Env
import numpy as np
import matplotlib.pyplot as plt

def observation_probability(env: Env, i: int, j: int, observation: list[int], epsilon: float) -> float:
    """
    Helper function to compute the emission probabiility of observing the sensor reading ``observation``
    in state (i, j) of the environment.
    
    Hint: You may want to call env.get_neighbors(i, j) to get the neighbors of (i, j).
    """
    # TODO: compute the emission probability
    # Get the true sensor reading for the state
    node = env._nodes[(i, j)]
    sensorReading = node._true_sensor_reading()

    # Compute probability for each sensor
    prob = 1.0
    for t, o in zip(sensorReading, observation):
        if t == o:
            prob *= (1 - epsilon)
        else:
            prob *= epsilon

    return prob

def viterbi(observations: list[list[int]], epsilon: float) -> np.ndarray:
    """
    Params: 
    observations: a list of observations of size (T, 4) where T is the number of observations and
    1. observations[t][0] is the reading of the left sensor at timestep t
    2. observations[t][1] is the reading of the right sensor at timestep t
    3. observations[t][2] is the reading of the up sensor at timestep t
    4. observations[t][3] is the reading of the down sensor at timestep t
    epsilon: the probability of a single sensor failing

    Return: a list of predictions for the agent's true hidden states.
    The expected output is a numpy array of shape (T, 2) where 
    1. (predictions[t][0], predictions[t][1]) is the prediction for the state at timestep t
    """
    # TODO: implement the viterbi algorithm
    T = len(observations)  # Number of timesteps
    rows, cols = env._rows, env._columns

    # Initialize DP table and backpointer
    dp = np.zeros((T, rows, cols))
    backPointer = np.zeros((T, rows, cols, 2), dtype=int)  # e.g. backPointer[5,2,3]=[1,3]

    # Initialization step
    for i in range(rows):
        for j in range(cols):
            dp[0, i, j] = observation_probability(env, i, j, observations[0], epsilon) / (rows * cols)  # Sum to 1

    # Recursion step
    for t in range(1, T):
        for i in range(rows):
            for j in range(cols):
                maxProb = -69
                bestPrevState = None

                for ni, nj in env.get_neighbors(i, j):  # Transition from neighbors
                    prob = dp[t - 1, ni, nj] / len(env.get_neighbors(ni, nj))
                    if prob > maxProb:
                        maxProb = prob
                        bestPrevState = (ni, nj)

                dp[t, i, j] = maxProb * observation_probability(env, i, j, observations[t], epsilon)
                backPointer[t, i, j] = bestPrevState

    # Backtrace
    predictions = np.zeros((T, 2), dtype=int)
    finalState = np.unravel_index(dp[T - 1].argmax(), dp[T - 1].shape)
    predictions[T - 1] = finalState
    for t in range(T - 2, -1, -1):
        predictions[t] = backPointer[t + 1, predictions[t + 1, 0], predictions[t + 1, 1]]

    return predictions
    

if __name__ == '__main__':
    random.seed(12345)
    rows, cols = 16, 16 # dimensions of the environment
    openness = 0.3 # some hyperparameter defining how "open" an environment is
    traj_len = 100 # number of observations to collect, i.e., number of times to call env.step()
    num_traj = 100 # number of trajectories to run per epsilon

    env = Env(rows, cols, openness)
    env.plot_env() # the environment layout should be saved to env_layout.png

    plt.clf()
    """
    The following loop simulates num_traj trajectories for each value of epsilon.
    Since there are 6 values of epsilon being tried here, a total of 6 * num_traj
    trajectories are generated.
    
    For reference, the staff solution takes < 3 minutes to run.
    """
    for epsilon in [0.0, 0.05, 0.1, 0.2, 0.25, 0.5]:
        env.set_epsilon(epsilon)
        
        accuracies = []
        for _ in range(num_traj):
            env.init_env()

            observations = []
            for i in range(traj_len):
                obs = env.step()
                observations.append(obs)

            predictions = viterbi(observations, epsilon)

            accuracies.append(env.compute_accuracy(predictions))
        plt.plot(np.mean(accuracies, axis=0), label=f"$ϵ$={epsilon}")
        plt.xlabel("Number of observations")
        plt.ylabel("Accuracy")
        plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
        plt.savefig("accuracies.png", bbox_inches='tight')
