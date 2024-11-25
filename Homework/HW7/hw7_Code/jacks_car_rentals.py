"""
Author: Aryan Jain
"""

import numpy as np
from scipy.stats import poisson
from itertools import product
from functools import cache
import matplotlib.pyplot as plt

@cache
def pmf(k: int, lam: float):
    """
    PMF = probability mass function.
    Computes P(X = k) where X ~ Poisson(lam).
    """
    return poisson.pmf(k, lam)

@cache
def cdf(k: int, lam: float):
    """
    CDF = cumulative distribution function.
    Computes P(X <= k) where X ~ Poisson(lam).
    """
    return poisson.cdf(k, lam)

@cache
def sf(k: int, lam: float):
    """
    SF = survival function = 1 - CDF.
    Computes P(X > k) where X ~ Poisson(lam).
    """
    return poisson.sf(k, lam)

@cache
def expectation(lam: float, c: int):
    """
    Suppose X ~ Poisson(lam).
    This function computes E[min(X, c)].
    """
    return sum(k * pmf(k, lam) for k in range(c + 1)) + (1 - cdf(c, lam)) * c

@cache
def transition_prob(state: tuple[int], action: int, next_state: tuple[int]) -> float:
    """
    Computes p(next_state | state, action), i.e., the probability of reaching 
    (next_state[0], next_state[1]) cars at location 1 and 2, respectively, at the end of the 
    next day given that we start at (state[0], state[1]) cars at the end of the current day and 
    move `action` cars overnight.

    Fill in the TODOs. You may find the helper functions above very helpful. In fact, you should use 
    them to compute the various probabilities involved in this function since the helper functions
    are cached and will speed up your computation.

    Note: this function assumes that `state` and `next_state` are tuples instead of numpy arrays.
    This is because we are caching this function and numpy arrays are not hashable!
    """

    # Cars at each location at the end of the current day
    loc_1_curr, loc_2_curr = state
    # We can't move more cars than we have at each location so we return 0 probability in this case
    if loc_1_curr < action or loc_2_curr < -action:
        # Yes, this is technically undefined behavior since we are assigning P(s' | s, a) = 0 for all s' and 
        # the axioms of probability would dictate that we get sum_{s'} P(s' | s, a) = 1. However, this can 
        # equivalently be viewed as restricting the action space to only the valid actions that can be taken 
        # from the current state, i.e., letting the action space be dependent on the state (this is allowed in 
        # the MDP framework).
        return 0
    # Cars at each location at the START of the next day assuming `action` cars are moved overnight
    loc_1_start, loc_2_start = loc_1_curr - action, loc_2_curr + action
    # The number of cars at each location is capped at MAX_CARS_LOC_1 and MAX_CARS_LOC_2
    # The excess cars are removed from the system.
    loc_1_start = min(loc_1_start, MAX_CARS_LOC_1)
    loc_2_start = min(loc_2_start, MAX_CARS_LOC_2)
    # Cars we want at each location by the END of the next day
    loc_1_end, loc_2_end = next_state

    # Handle the probability of going from `loc_1_start` to `loc_1_end` first
    prob_loc_1 = 0

    # Split into 2 cases:
    # Case 1: we can only rent up to (<=) `loc_1_start` cars the next day
    # To reach `loc_1_end` cars by the end of the day, how many cars do we need to rent out at minimum?
    # Hint: what happens when `loc_1_start` > `loc_1_end`? what about `loc_1_start` < `loc_1_end`?
    min_cars_to_rent = max(..., ...) # TODO
    for rented in range(min_cars_to_rent, loc_1_start + 1):
        # Now, find out the number of cars that we want to see returned back to reach `loc_1_end` cars by the 
        # end of the day, given that we rented out `rented` cars at location 1:
        returned = ... # TODO
        # We again have two cases:
        if loc_1_end < MAX_CARS_LOC_1:
            # Case 1.1: `loc_1_end` < MAX_CARS_LOC_1
            # Here, we need exactly `returned` cars to be returned. 
            # Compute the probability of seeing exactly `returned` cars returned to location 1.
            prob_returned = ... # TODO
        else:
            # Case 1.2: `loc_1_end` == MAX_CARS_LOC_1
            # Here, we need at least `returned` cars to be returned.
            # The number of cars will be capped at MAX_CARS_LOC_1 and the excess is removed from the system.
            # Compute the probability of seeing >= `returned` cars returned to location 1.
            prob_returned = ... # TODO
        # Compute the probability of seeing exactly `rented` cars being rented out.
        prob_rented = ... # TODO
        # Combine the two probabilities above. Note that the rentals and returns are independent of each other.
        prob_loc_1 += ... # TODO
    # Case 2: we get more rental requests than we have cars available at location 1.
    # Of course, we can only rent out `loc_1_start` cars before running out but that doesn't affect the number 
    # of requests that come in. 
    # Since we are renting out all of the cars available at location 1, we would need exactly `loc_1_end` cars 
    # to be returned to reach `loc_1_end` cars by the end of the day.
    # Compute the probability of this event (more than `loc_1_start` rental requests and exactly `loc_1_end` 
    # returns) happening.
    prob_loc_1 += ... # TODO

    # We repeat the same process for location 2
    prob_loc_2 = 0

    # Split into 2 cases:
    # Case 1: we can only rent up to (<=) `loc_2_start` cars the next day
    # To reach `loc_2_end` cars by the end of the day, how many cars do we need to rent out at minimum?
    # Hint: what happens when `loc_2_start` > `loc_2_end`? what about `loc_2_start` < `loc_2_end`?
    min_cars_to_rent = max(..., ...) # TODO
    for rented in range(min_cars_to_rent, loc_2_start + 1):
        # Now, find out the number of cars that we want to see returned back to reach `loc_2_end` cars by the 
        # end of the day, given that we rented out `rented` cars at location 2:
        returned = ... # TODO
        # We again have two cases:
        if loc_2_end < MAX_CARS_LOC_2:
            # Case 1.1: `loc_2_end` < MAX_CARS_LOC_2
            # Here, we need exactly `returned` cars to be returned. 
            # Compute the probability of seeing exactly `returned` cars returned to location 2.
            prob_returned = ... # TODO
        else:
            # Case 1.2: `loc_2_end` == MAX_CARS_LOC_2
            # Here, we need at least `returned` cars to be returned.
            # The number of cars will be capped at MAX_CARS_LOC_2 and the excess is removed from the system.
            # Compute the probability of seeing >= `returned` cars returned to location 2.
            prob_returned = ... # TODO
        # Compute the probability of seeing exactly `rented` cars being rented out.
        prob_rented = ... # TODO
        # Combine the two probabilities above. Note that the rentals and returns are independent of each other.
        prob_loc_2 += ... # TODO
    # Case 2: we get more rental requests than we have cars available at location 2.
    # Of course, we can only rent out `loc_2_start` cars before running out but that doesn't affect the number 
    # of requests that come in. 
    # Since we are renting out all of the cars available at location 1, we would need exactly `loc_2_end` cars 
    # to be returned to reach `loc_2_end` cars by the end of the day.
    # Compute the probability of this event (more than `loc_2_start` rental requests and exactly `loc_2_end` 
    # returns) happening.
    prob_loc_2 += ... # TODO

    # We take the product of the two probabilities since each location operates independently of the other.
    return prob_loc_1 * prob_loc_2

@cache
def expected_rewards(state: tuple[int], action: int, add_non_linearity: bool = True) -> float:
    """
    Helper function to compute E[r | state, action].
    Implement this function and use it as a subroutine for your q-function computation.

    `state`: a 2-tuple (loc_1_cars, loc_2_cars).
    `action`: a number between -MAX_CARS_MOVED and MAX_CARS_MOVED (inclusive) where a
        1. positive number means moving cars from location 1 to location 2
        2. negative number means moving cars from location 2 to location 1
    `add_non_linearity`: a boolean indicating whether we add the non-linearities from part (d) to the reward 
    function; This function should have 2 possible return values depending on what `add_non_linearity` is set 
    to. You can ignore this flag for parts (b) and (c).

    Note: this function assumes that `state` is a tuple instead of a numpy array.
    This is because we are caching this function and numpy arrays are not hashable!
    """
    # TODO
    pass


def q(
    state: np.ndarray, 
    action: int, 
    V: np.ndarray, 
    state_space: np.ndarray, 
    add_non_linearity: bool
) -> float:
    """
    Helper function to compute the q-value sum_{s', r} p(s', r | s, a) * (r + gamma * V(s')).
    Again, as before, s is `state` and a is `action`.
    Implement this function and use it as a subroutine for policy evaluation + improvement and value iteration.
    Hint: you may find the decomposition of the q-value from part (a) helpful.
    """
    # TODO
    pass

def policy_evaluation(
    policy: np.ndarray, 
    V: np.ndarray, 
    state_space: np.ndarray,
    add_non_linearity: bool, 
    threshold: float
) -> None:
    """
    Runs policy evaluation using the given `policy` and updates `V` in-place.
    You should run the policy evaluation loop until the maximum change in `V` is less than `threshold`.
    """
    # TODO
    pass

def policy_improvement(
    policy: np.ndarray, 
    V: np.ndarray, 
    state_space: np.ndarray, 
    action_space: np.ndarray,
    add_non_linearity: bool, 
    threshold: float
) -> None:
    """
    Runs policy improvement on the given `policy` and updates it in-place.
    As shown in lecture, we update the policy greedily by choosing the action that maximizes the q-value.
    You should run the policy improvement loop until the policy is stable.
    Remember to call policy_evaluation() as a subroutine!

    For reference, the staff solution takes ~1.5 minutes to run.
    """
    # TODO
    pass

def value_iteration(
    policy: np.ndarray,
    V: np.ndarray, 
    state_space: np.ndarray, 
    action_space: np.ndarray, 
    add_non_linearity: bool, 
    threshold: float
) -> None:
    """
    Run value iteration to compute the optimal value function.
    You should run the value iteration loop until the maximum change in `V` is less than `threshold`.
    Finally, set `policy` the optimal policy that greedily picks the q-value maximizing action in each state.

    For reference, the staff solution ~2 minutes to run.
    """
    # TODO
    pass

def init_policy() -> np.ndarray:
    return np.zeros((MAX_CARS_LOC_1 + 1, MAX_CARS_LOC_2 + 1)).astype(np.int32)

def init_value() -> np.ndarray:
    return np.zeros((MAX_CARS_LOC_1 + 1, MAX_CARS_LOC_2 + 1))

def plot_policy_value(policy: np.ndarray, V: np.ndarray, part: str) -> None:
    cmap = plt.get_cmap('RdBu', 11)
    mat = plt.matshow(policy.T, cmap=cmap, vmin=-5.5, vmax=5.5)
    plt.gca().invert_yaxis()
    plt.gca().xaxis.set_ticks_position('bottom')
    plt.colorbar(mat, ticks=np.arange(-5, 6))
    plt.xlabel("Number of cars at location 1")
    plt.ylabel("Number of cars at location 2")
    plt.savefig(f'policy_{part}.png', dpi=300, bbox_inches='tight')
    plt.clf()

    plt.matshow(V.T)
    plt.gca().invert_yaxis()
    plt.gca().xaxis.set_ticks_position('bottom')
    plt.colorbar()
    plt.xlabel("Number of cars at location 1")
    plt.ylabel("Number of cars at location 2")
    plt.savefig(f'V_{part}.png', dpi=300, bbox_inches='tight')
    plt.clf()
    
if __name__ == "__main__":
    MAX_CARS_LOC_1 = 20 # maximum number of cars at location 1
    MAX_CARS_LOC_2 = 20 # maximum number of cars at location 2
    MAX_CARS_MOVED = 5 # maximum number of cars that can be moved in one night

    RENTAL_REWARD = 10  # reward for renting out a car
    MOVE_COST = 2 # cost for moving a car overnight

    LOC_1_RENTAL_LAMBDA = 3
    LOC_1_RETURN_LAMBDA = 3
    LOC_2_RENTAL_LAMBDA = 4
    LOC_2_RETURN_LAMBDA = 2
    
    GAMMA = 0.9 # discount factor

    THRESHOLD = 1e-2 # the convergence threshold used for each of the algorithms above
    # You may want to set the threshold to something high when debugging since it will allow each algorithm 
    # to terminate early.
    
    # constants for part (d) and (e)
    FREE_MOVES = 1 # number of free moves from location 1 to 2 that Jack's employee can make
    STORAGE_CAPACITY = 10 # maximum number of cars that can be stored at each location overnight for free
    STORAGE_COST = 4 # cost incurred for the second parking lot at a given location

    state_space = np.array(list(product(range(MAX_CARS_LOC_1 + 1), range(MAX_CARS_LOC_2 + 1)))).astype(np.int8)
    action_space = np.arange(-MAX_CARS_MOVED, MAX_CARS_MOVED + 1).astype(np.int8)

    policy = init_policy()
    V = init_value()    
    policy_improvement(policy, V, state_space, action_space, add_non_linearity=False, threshold=THRESHOLD)
    plot_policy_value(policy, V, "part(b)")

    # ===================================================================================

    policy = init_policy()
    V = init_value()
    value_iteration(policy, V, state_space, action_space, add_non_linearity=False, threshold=THRESHOLD)
    plot_policy_value(policy, V, "part(c)")

    # ===================================================================================

    policy = init_policy()
    V = init_value()
    policy_improvement(policy, V, state_space, action_space, add_non_linearity=True, threshold=THRESHOLD)
    plot_policy_value(policy, V, "part(d)")

    # ===================================================================================

    policy = init_policy()
    V = init_value()
    value_iteration(policy, V, state_space, action_space, add_non_linearity=True, threshold=THRESHOLD)
    plot_policy_value(policy, V, "part(e)")