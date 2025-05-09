from typing import Set, List, Union, Tuple
import numpy as np
from sortedcontainers import SortedDict

def initialize_pq(gnd: List[int], f_diff, ind_add_oracle, num_sol: int, knapsack_constraints: Union[np.ndarray, None]) -> Tuple[SortedDict, int, int]:
    num_oracle = 0
    num_fun = 0

    # Sử dụng SortedDict để thay thế PriorityQueue với thứ tự ngược (max heap)
    pq = SortedDict()
    empty_set = set()

    for elm in gnd:
        num_oracle += 1
        if ind_add_oracle(elm, empty_set) and feasible_knapsack_elm(elm, knapsack_constraints):
            density = get_density(elm, knapsack_constraints)
            prev_size = 0
            gain = f_diff(elm, empty_set)
            num_fun += 1

            for sol_ind in range(1, num_sol + 1):
                key = (elm, sol_ind, prev_size, density)
                pq[float(-gain)] = key  # Chuyển đổi -gain thành float để tránh lỗi với numpy.float64

    return pq, num_oracle, num_fun

# Knapsack functionality
def feasible_knapsack_elm(elm: int, knapsack_constraints: Union[np.ndarray, None]) -> bool:
    if knapsack_constraints is None:
        return True
    else:
        return np.all(knapsack_constraints[:, elm] <= 1.0)

def get_density(elm: int, knapsack_constraints: Union[np.ndarray, None]) -> float:
    if knapsack_constraints is None:
        return 0.0
    else:
        return np.sum(knapsack_constraints[:, elm])

def knapsack_feasible_to_add(elm: int, set_to_update: int, sol_costs: Union[np.ndarray, None], knapsack_constraints: Union[np.ndarray, None]) -> bool:
    if knapsack_constraints is None:
        return True
    else:
        return np.all(sol_costs[:, set_to_update] + knapsack_constraints[:, elm] <= 1)

def ignorable_knapsack(knapsack_constraints: Union[np.ndarray, None]) -> bool:
    if knapsack_constraints is None:
        return True
    else:
        m, n_k = knapsack_constraints.shape
        return np.all(np.sum(knapsack_constraints, axis=1) <= 1)

def num_knapsack(knapsack_constraints: Union[np.ndarray, None]) -> int:
    if knapsack_constraints is None:
        return 0
    else:
        return knapsack_constraints.shape[0]

def init_knapsack_costs(num_sol: int, knapsack_constraints: Union[np.ndarray, None]) -> Union[np.ndarray, None]:
    if knapsack_constraints is None:
        return None
    else:
        m, n = knapsack_constraints.shape
        sol_costs = np.zeros((m, num_sol))
        return sol_costs

def update_sol_costs(elm_to_add: int, set_to_update: int, sol_costs: Union[np.ndarray, None], knapsack_constraints: Union[np.ndarray, None]) -> None:
    if knapsack_constraints is not None:
        sol_costs[:, set_to_update] += knapsack_constraints[:, elm_to_add]

def dimension_check(gnd: List[int], knapsack_constraints: Union[np.ndarray, None]) -> bool:
    if knapsack_constraints is None:
        return True
    else:
        n = max(gnd)
        m, n_k = knapsack_constraints.shape
        return n <= n_k

# Independence constraints
def intersection_ind_oracle(elm: int, sol: Set[int], *ind_list) -> bool:
    for ind_add_oracle in ind_list:
        if not ind_add_oracle(elm, sol):
            return False
    return True

# Printing functions
def myshow(x) -> None:
    print(x)

def printset(sol: Set[int]) -> None:
    if len(sol) == 0:
        print("{ }")
    else:
        sol_list = sorted(list(sol))
        n = len(sol_list)
        print("{ ", end="")
        for i in range(n - 1):
            elm = sol_list[i]
            print(f"{elm}, ", end="")
        print(f"{sol_list[n-1]}", end="")

def printlnset(sol: Set[int]) -> None:
    printset(sol)
    print()