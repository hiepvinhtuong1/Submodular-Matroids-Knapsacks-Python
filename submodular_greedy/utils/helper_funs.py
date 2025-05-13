import heapq
from typing import List, Set, Tuple, Union, Dict
import numpy as np

def initialize_pq(gnd: List[int], f_diff, ind_add_oracle, num_sol: int, knapsack_constraints: Union[np.ndarray, None]) -> Tuple[List, int, int]:
    """
    Initialize a priority queue with initial marginal gains.
    Returns: (priority queue, number of oracle calls, number of function evaluations)
    """
    num_oracle = 0
    num_fun = 0

    # Priority queue as a list of tuples (priority, (elm, sol_ind, staleness, density))
    # heapq in Python is a min-heap, so we use negative gain for max-heap behavior
    pq = []
    
    emptyset = set()
    for elm in gnd:
        num_oracle += 1
        if ind_add_oracle(elm, emptyset) and feasible_knapsack_elm(elm, knapsack_constraints):
            density = get_density(elm, knapsack_constraints)
            prev_size = 0
            gain = f_diff(elm, emptyset)
            num_fun += 1

            for sol_ind in range(1, num_sol + 1):
                heapq.heappush(pq, (-gain, (elm, sol_ind, prev_size, density)))

    return pq, num_oracle, num_fun

# Knapsack Functionality
def feasible_knapsack_elm(elm: int, knapsack_constraints: Union[np.ndarray, None]) -> bool:
    if knapsack_constraints is None:
        return True
    return np.all(knapsack_constraints[:, elm-1] <= 1.0)

def get_density(elm: int, knapsack_constraints: Union[np.ndarray, None]) -> float:
    if knapsack_constraints is None:
        return 0.0
    return np.sum(knapsack_constraints[:, elm-1])

def knapsack_feasible_to_add(elm: int, set_to_update: int, sol_costs: Union[np.ndarray, None], knapsack_constraints: Union[np.ndarray, None]) -> bool:
    if knapsack_constraints is None:
        return True
    return np.all(sol_costs[:, set_to_update-1] + knapsack_constraints[:, elm-1] <= 1)

def ignorable_knapsack(knapsack_constraints: Union[np.ndarray, None]) -> bool:
    if knapsack_constraints is None:
        return True
    m, _ = knapsack_constraints.shape
    return np.all(np.sum(knapsack_constraints, axis=1) <= 1)

def num_knapsack(knapsack_constraints: Union[np.ndarray, None]) -> int:
    if knapsack_constraints is None:
        return 0
    return knapsack_constraints.shape[0]

def init_knapsack_costs(num_sol: int, knapsack_constraints: Union[np.ndarray, None]) -> Union[np.ndarray, None]:
    if knapsack_constraints is None:
        return None
    m, _ = knapsack_constraints.shape
    return np.zeros((m, num_sol))

def update_sol_costs(elm_to_add: int, set_to_update: int, sol_costs: Union[np.ndarray, None], knapsack_constraints: Union[np.ndarray, None]) -> None:
    if knapsack_constraints is not None:
        sol_costs[:, set_to_update-1] += knapsack_constraints[:, elm_to_add-1]

def dimension_check(gnd: List[int], knapsack_constraints: Union[np.ndarray, None]) -> bool:
    if knapsack_constraints is None:
        return True
    n = max(gnd)
    _, n_k = knapsack_constraints.shape
    return n <= n_k

# Independence Constraints
def intersection_ind_oracle(elm: int, sol: Set[int], *ind_list) -> bool:
    for ind_add_oracle in ind_list:
        if not ind_add_oracle(elm, sol):
            return False
    return True

# Printing
def printset(sol: Set[int]) -> None:
    if not sol:
        print("{ }")
    else:
        sol_list = sorted(list(sol))
        n = len(sol)
        print("{ ", end="")
        for i in range(n-1):
            elm = sol_list[i]
            print(f"{elm}, ", end="")
        print(f"{sol_list[n-1]} ", end="")

def printlnset(sol: Set[int]) -> None:
    printset(sol)
    print("")