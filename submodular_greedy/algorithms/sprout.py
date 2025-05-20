import random
import math
import cupy as cp
from typing import List, Set, Tuple, Union
from submodular_greedy.utils.helper_funs import initialize_pq, ignorable_knapsack, num_knapsack
from itertools import combinations

def density_search_for_sprout(param_c: int, fA_value: float, pq: List, num_sol: int, beta_scaling: float, gamma_scaling: float, delta: float, f_diff, ind_add_oracle, knapsack_constraints: Union[cp.ndarray, None], epsilon: float, opt_size_ub: int, verbose: bool = True, mu: float = 1.0) -> Tuple[Set[int], float, int, int]:
    # Sửa đoạn code lấy max_gain
    if not pq:  # Kiểm tra nếu pq rỗng
        return set(), 0.0, 0, 0

    neg_max_gain, _ = pq[0]  # Lấy phần tử đầu tiên của pq
    max_gain = -neg_max_gain  # Phủ định để có giá trị max_gain

    num_fun = 0
    num_oracle = 0

    lower_density_ratio = beta_scaling * max_gain * 1
    upper_density_ratio = beta_scaling * max_gain * opt_size_ub

    best_sol = set()
    best_f_val = 0
    best_density_ratio = -1

    iter = 1
    while upper_density_ratio > (1 + delta) * lower_density_ratio:
        density_ratio = math.sqrt(lower_density_ratio * upper_density_ratio)

        from submodular_greedy.algorithms.simultaneous_greedys import simultaneous_greedy_alg
        sol, fval, num_f, num_or, knap_reject = simultaneous_greedy_alg(pq.copy(), num_sol, epsilon, f_diff, ind_add_oracle, knapsack_constraints, density_ratio + gamma_scaling * fA_value / param_c, opt_size_ub)

        if fval > best_f_val:
            best_f_val = fval
            best_sol = sol.copy()
            best_density_ratio = density_ratio

        num_fun += num_f
        num_oracle += num_or

        if knap_reject:
            lower_density_ratio = density_ratio - (mu - 1.0) * (density_ratio - lower_density_ratio) / mu
        else:
            upper_density_ratio = density_ratio + (mu - 1.0) * (upper_density_ratio - density_ratio) / mu

        iter += 1

    # Đồng bộ GPU
    cp.cuda.Stream.null.synchronize()
    return best_sol, best_f_val, num_fun, num_oracle

def init_sgs_params_for_sprout(num_sol: int, k: int, extendible: bool, monotone: bool, knapsack_constraints: Union[cp.ndarray, None], epsilon: float) -> Tuple[int, bool, float, float]:
    if ignorable_knapsack(knapsack_constraints):
        run_density_search = False
        m = 0
    else:
        run_density_search = True
        m = num_knapsack(knapsack_constraints)
        assert k > 0, "k must be specified for density search with knapsack constraints"

    if monotone:
        if num_sol == 0:
            num_sol = 1

        if extendible:
            p = max(num_sol - 1, k)
        else:
            p = k + num_sol - 1

        beta_scaling = 0.0005
    else:
        if extendible:
            M = max(int(math.ceil(math.sqrt(1 + 2 * m))), k)
            if num_sol == 0:
                num_sol = M + 1
            p = M
        else:
            if num_sol == 0:
                num_sol = int(math.floor(2 + math.sqrt(k + 2 * m + 2)))
            p = k + num_sol - 1

        beta_scaling = 0.0005
        gamma_scaling = 1e-6

    return num_sol, run_density_search, beta_scaling, gamma_scaling

def main_part_sprout(param_c: int, fA_value: float, gnd: List[int], f_diff, ind_add_oracle, num_sol: int = 0, k: int = 0, knapsack_constraints: Union[cp.ndarray, None] = None, extendible: bool = True, monotone: bool = False, epsilon: float = 0.0, opt_size_ub: int = None, verbose_lvl: int = 1, mu: float = 1.0) -> Tuple[Set[int], float, int, int]:
    if not gnd:
        best_sol = set()
        best_f_val = 0
        num_fun = 0
        num_oracle = 0
        return best_sol, best_f_val, num_fun, num_oracle

    assert (num_sol > 0) or (k > 0), "At least num_sol or k needs to be specified"
    assert ignorable_knapsack(knapsack_constraints) or (epsilon > 0), "Specify non-zero epsilon for density search with knapsack constraints"
    assert ignorable_knapsack(knapsack_constraints) or (k > 0), "Specify k for density search with knapsack constraints"
    assert 0.0 <= epsilon <= 1.0, "Epsilon must be in [0, 1]"

    if opt_size_ub is None:
        opt_size_ub = len(gnd)

    num_sol, run_density_search, beta_scaling, gamma_scaling = init_sgs_params_for_sprout(num_sol, k, extendible, monotone, knapsack_constraints, epsilon)
    pq, num_fun, num_oracle = initialize_pq(gnd, f_diff, ind_add_oracle, num_sol, knapsack_constraints)

    if not pq:
        best_sol = set()
        best_f_val = 0
        return best_sol, best_f_val, num_fun, num_oracle

    if run_density_search:
        delta = epsilon
        best_sol, best_f_val, num_f, num_or = density_search_for_sprout(param_c, fA_value, pq, num_sol, beta_scaling, gamma_scaling, delta, f_diff, ind_add_oracle, knapsack_constraints, epsilon, opt_size_ub, verbose=verbose_lvl >= 2, mu=mu)
    else:
        density_ratio = 0.0
        from submodular_greedy.algorithms.simultaneous_greedys import simultaneous_greedy_alg
        best_sol, best_f_val, num_f, num_or, _ = simultaneous_greedy_alg(pq, num_sol, epsilon, f_diff, ind_add_oracle, knapsack_constraints, density_ratio, opt_size_ub, verbose=verbose_lvl >= 2)

    num_fun += num_f
    num_oracle += num_or

    # Đồng bộ GPU
    cp.cuda.Stream.null.synchronize()
    return best_sol, best_f_val, num_fun, num_oracle