from typing import Set, List, Union, Tuple
import numpy as np
import time
from sortedcontainers import SortedDict
from ..utils.helper_funs import (
    init_knapsack_costs, knapsack_feasible_to_add, update_sol_costs,
    ignorable_knapsack, num_knapsack, dimension_check, initialize_pq
)

def simultaneous_greedy_alg(pq: SortedDict, num_sol: int, epsilon: float, f_diff, ind_add_oracle,
                            knapsack_constraints: Union[np.ndarray, None], density_ratio: float,
                            opt_size_ub: int, verbose: bool = False) -> Tuple[Set[int], float, int, int, bool]:
    assert num_sol > 0
    assert 0 <= epsilon <= 1

    if len(pq) == 0:
        return set(), 0.0, 0, 0, False

    sol_list = [set() for _ in range(num_sol)]
    sol_vals_list = [0.0 for _ in range(num_sol)]
    sol_costs = init_knapsack_costs(num_sol, knapsack_constraints)

    num_fun = 0
    num_oracle = 0
    knap_reject = False

    first_key = next(iter(pq))
    best_elm_info = pq[first_key]
    max_gain = -first_key[0]  # first element's -f_gain

    threshold = (1 - epsilon) * max_gain
    min_threshold = epsilon * max_gain / opt_size_ub
    prev_threshold = threshold
    iter = 1

    while threshold > min_threshold and len(pq) > 0:
        key, (elm, sol_ind, prev_size, density) = next(iter(pq.items()))
        del pq[key]
        sol_ind -= 1  # Adjust for 0-based index

        if prev_size == len(sol_list[sol_ind]):
            if knapsack_feasible_to_add(elm, sol_ind, sol_costs, knapsack_constraints):
                sol_list[sol_ind].add(elm)
                sol_vals_list[sol_ind] += -key[0]
                update_sol_costs(elm, sol_ind, sol_costs, knapsack_constraints)

                keys_to_remove = [k for k, v in pq.items() if v[0] == elm]
                for k in keys_to_remove:
                    del pq[k]

                iter += 1

            if -key[0] < threshold and len(pq) > 0:
                prev_threshold = threshold
                next_key = next(iter(pq))
                threshold = min((1 - epsilon) * threshold, -next_key[0])
        else:
            num_oracle += 1
            if ind_add_oracle(elm, sol_list[sol_ind]):
                num_fun += 1
                f_gain = f_diff(elm, sol_list[sol_ind])

                if f_gain > density_ratio * density:
                    prev_size = len(sol_list[sol_ind])
                    unique_id = time.time_ns()
                    pq[(-f_gain, unique_id)] = (elm, sol_ind + 1, prev_size, density)
                else:
                    knap_reject = True

    sol_list.append({best_elm_info[0]})
    sol_vals_list.append(max_gain)

    best_sol_ind = np.argmax(sol_vals_list)
    best_sol = sol_list[best_sol_ind]
    best_f_val = sol_vals_list[best_sol_ind]

    return best_sol, best_f_val, num_fun, num_oracle, knap_reject


def density_search(pq: SortedDict, num_sol: int, beta_scaling: float, delta: float,
                   f_diff, ind_add_oracle, knapsack_constraints: Union[np.ndarray, None],
                   epsilon: float, opt_size_ub: int, verbose: bool = True) -> Tuple[Set[int], float, int, int]:
    if len(pq) == 0:
        return set(), float('-inf'), 0, 0

    first_key = next(iter(pq))
    max_gain = -first_key[0]

    num_fun = 0
    num_oracle = 0
    lower_density_ratio = beta_scaling * max_gain * 1
    upper_density_ratio = beta_scaling * max_gain * opt_size_ub

    best_sol = None
    best_f_val = float('-inf')

    while upper_density_ratio > (1 + delta) * lower_density_ratio:
        density_ratio = np.sqrt(lower_density_ratio * upper_density_ratio)

        copied_pq = SortedDict(pq)
        sol, fval, num_f, num_or, knap_reject = simultaneous_greedy_alg(
            copied_pq, num_sol, epsilon, f_diff, ind_add_oracle,
            knapsack_constraints, density_ratio, opt_size_ub, verbose
        )

        if fval > best_f_val:
            best_f_val = fval
            best_sol = sol

        num_fun += num_f
        num_oracle += num_or

        if knap_reject:
            upper_density_ratio = density_ratio
        else:
            lower_density_ratio = density_ratio

    return best_sol, best_f_val, num_fun, num_oracle


def init_sgs_params(num_sol: int, k: int, extendible: bool, monotone: bool,
                    knapsack_constraints: Union[np.ndarray, None], epsilon: float) -> Tuple[int, bool, float]:
    if ignorable_knapsack(knapsack_constraints):
        run_density_search = False
        m = 0
    else:
        run_density_search = True
        m = num_knapsack(knapsack_constraints)
        assert k > 0

    if monotone:
        if num_sol == 0:
            num_sol = 1
        if extendible:
            p = max(num_sol - 1, k)
        else:
            p = k + num_sol - 1
        beta_scaling = 2 * (1 - epsilon) ** 2 / (p + 1 + 2 * m)
    else:
        if extendible:
            M = max(int(np.ceil(np.sqrt(1 + 2 * m))), k)
            if num_sol == 0:
                num_sol = M + 1
            p = M
        else:
            if num_sol == 0:
                num_sol = int(np.floor(2 + np.sqrt(k + 2 * m + 2)))
            p = k + num_sol - 1
        beta_scaling = 2 * (1 - epsilon) * (1 - 1 / num_sol - epsilon) / (p + 1 + 2 * m)

    return num_sol, run_density_search, beta_scaling


def simultaneous_greedys(gnd: List[int], f_diff, ind_add_oracle, num_sol: int = 0, k: int = 0,
                         knapsack_constraints: Union[np.ndarray, None] = None, extendible: bool = False,
                         monotone: bool = False, epsilon: float = 0.0, opt_size_ub: int = None,
                         verbose_lvl: int = 1) -> Tuple[Set[int], float, int, int]:
    start_time = time.time()

    assert (num_sol > 0) or (k > 0), "At least num_sol or k need to be specified."
    assert dimension_check(gnd, knapsack_constraints), "More elements in knapsack constraints than in the ground set."
    assert ignorable_knapsack(knapsack_constraints) or (epsilon > 0), "Non-zero epsilon required for density search with knapsack constraints."
    assert ignorable_knapsack(knapsack_constraints) or (k > 0), "k required for density search with knapsack constraints."
    assert 0.0 <= epsilon <= 1.0, "Epsilon must be in range [0, 1]"

    if opt_size_ub is None:
        opt_size_ub = len(gnd)

    num_sol, run_density_search, beta_scaling = init_sgs_params(
        num_sol, k, extendible, monotone, knapsack_constraints, epsilon
    )

    pq, num_fun, num_oracle = initialize_pq(gnd, f_diff, ind_add_oracle, num_sol, knapsack_constraints)

    if len(pq) == 0:
        return set(), float('-inf'), num_fun, num_oracle

    info_verbose = verbose_lvl >= 1
    alg_verbose = verbose_lvl >= 2

    if run_density_search:
        delta = epsilon
        best_sol, best_f_val, num_f, num_or = density_search(
            pq, num_sol, beta_scaling, delta, f_diff, ind_add_oracle,
            knapsack_constraints, epsilon, opt_size_ub, verbose=alg_verbose
        )
    else:
        best_sol, best_f_val, num_f, num_or, _ = simultaneous_greedy_alg(
            pq, num_sol, epsilon, f_diff, ind_add_oracle,
            knapsack_constraints, 0.0, opt_size_ub, verbose=alg_verbose
        )

    num_fun += num_f
    num_oracle += num_or
    end_time = time.time()

    if info_verbose:
        print(f"Simultaneous Greedys runtime: {end_time - start_time:.4f} seconds")

    return best_sol, best_f_val, num_fun, num_oracle