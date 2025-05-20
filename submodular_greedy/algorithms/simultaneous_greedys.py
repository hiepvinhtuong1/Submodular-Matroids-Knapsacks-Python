import time
import heapq
import math
import cupy as cp
from typing import List, Set, Tuple, Union
from submodular_greedy.utils.helper_funs import initialize_pq, knapsack_feasible_to_add, init_knapsack_costs, update_sol_costs, ignorable_knapsack, num_knapsack, dimension_check

def simultaneous_greedy_alg(pq: List, num_sol: int, epsilon: float, f_diff, ind_add_oracle, knapsack_constraints: Union[cp.ndarray, None], density_ratio: float, opt_size_ub: int, verbose: bool = False) -> Tuple[Set[int], float, int, int, bool]:
    assert num_sol > 0, "num_sol must be positive"
    assert 0 <= epsilon <= 1, "epsilon must be in [0, 1]"

    n = len(pq)
    sol_list = [set() for _ in range(num_sol)]
    sol_vals_list = [0.0 for _ in range(num_sol)]

    sol_costs = init_knapsack_costs(num_sol, knapsack_constraints)

    num_fun = 0
    num_oracle = 0
    knap_reject = False

    # Sửa đoạn code lấy max_gain
    if not pq:  # Kiểm tra nếu pq rỗng
        return set(), 0.0, num_fun, num_oracle, knap_reject

    neg_max_gain, best_elm_info = pq[0]  # Lấy phần tử đầu tiên của pq
    max_gain = -neg_max_gain  # Phủ định để có giá trị max_gain

    threshold = (1 - epsilon) * max_gain
    min_threshold = epsilon * max_gain / opt_size_ub
    prev_threshold = threshold

    iter = 1
    print_iter = True

    while (threshold > min_threshold) and pq:
        (neg_f_gain, (elm, sol_ind, prev_size, density)), pq = pq[0], pq[1:]
        f_gain = -neg_f_gain

        if prev_size == len(sol_list[sol_ind-1]):
            if knapsack_feasible_to_add(elm, sol_ind, sol_costs, knapsack_constraints):
                sol_list[sol_ind-1].add(elm)
                sol_vals_list[sol_ind-1] += f_gain
                update_sol_costs(elm, sol_ind, sol_costs, knapsack_constraints)

                pq = [(p, k) for p, k in pq if k[0] != elm]

                iter += 1
                print_iter = True

            if (f_gain < threshold) and pq:
                prev_threshold = threshold
                next_gain = -pq[0][0] if pq else float('-inf')
                threshold = min((1 - epsilon) * threshold, next_gain)

        else:
            num_oracle += 1
            if ind_add_oracle(elm, sol_list[sol_ind-1]):
                num_fun += 1
                f_gain = f_diff(elm, sol_list[sol_ind-1])

                if f_gain > density_ratio * density:
                    prev_size = len(sol_list[sol_ind-1])
                    heapq.heappush(pq, (-f_gain, (elm, sol_ind, prev_size, density)))
                else:
                    knap_reject = True

    sol_list.append({best_elm_info[0]})
    sol_vals_list.append(max_gain)

    best_sol_ind = cp.argmax(cp.array(sol_vals_list)).item()
    best_sol = sol_list[best_sol_ind]
    best_f_val = sol_vals_list[best_sol_ind]

    # Đồng bộ GPU
    cp.cuda.Stream.null.synchronize()
    return best_sol, best_f_val, num_fun, num_oracle, knap_reject

def density_search(pq: List, num_sol: int, beta_scaling: float, delta: float, f_diff, ind_add_oracle, knapsack_constraints: Union[cp.ndarray, None], epsilon: float, opt_size_ub: int, verbose: bool = True) -> Tuple[Set[int], float, int, int]:
    # Sửa đoạn code lấy max_gain
    if not pq:  # Kiểm tra nếu pq rỗng
        return set(), 0.0, 0, 0

    neg_max_gain, _ = pq[0]  # Lấy phần tử đầu tiên của pq
    max_gain = -neg_max_gain  # Phủ định để có giá trị max_gain

    num_fun = 0
    num_oracle = 0

    lower_density_ratio = beta_scaling * max_gain * 1
    upper_density_ratio = beta_scaling * max_gain * opt_size_ub

    best_sol = None
    best_f_val = float('-inf')

    iter = 1
    while upper_density_ratio > (1 + delta) * lower_density_ratio:
        density_ratio = math.sqrt(lower_density_ratio * upper_density_ratio)

        sol, fval, num_f, num_or, knap_reject = simultaneous_greedy_alg(pq.copy(), num_sol, epsilon, f_diff, ind_add_oracle, knapsack_constraints, density_ratio, opt_size_ub)

        if fval > best_f_val:
            best_f_val = fval
            best_sol = sol.copy()

        num_fun += num_f
        num_oracle += num_or

        if knap_reject:
            upper_density_ratio = density_ratio
        else:
            lower_density_ratio = density_ratio

        iter += 1

    # Đồng bộ GPU
    cp.cuda.Stream.null.synchronize()
    return best_sol, best_f_val, num_fun, num_oracle

def init_sgs_params(num_sol: int, k: int, extendible: bool, monotone: bool, knapsack_constraints: Union[cp.ndarray, None], epsilon: float) -> Tuple[int, bool, float]:
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

        beta_scaling = 2 * (1 - epsilon)**2 / (p + 1 + 2 * m)

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

        beta_scaling = 2 * (1 - epsilon) * (1 - 1 / num_sol - epsilon) / (p + 1 + 2 * m)

    return num_sol, run_density_search, beta_scaling

def simultaneous_greedys(gnd: List[int], f_diff, ind_add_oracle, num_sol: int = 0, k: int = 0, knapsack_constraints: Union[cp.ndarray, None] = None, extendible: bool = False, monotone: bool = False, epsilon: float = 0.0, opt_size_ub: int = None, verbose_lvl: int = 1) -> Tuple[Set[int], float, int, int]:
    start_time = time.time()

    assert (num_sol > 0) or (k > 0), "At least num_sol or k needs to be specified"
    assert dimension_check(gnd, knapsack_constraints), "More elements in knapsack constraints than in ground set"
    assert ignorable_knapsack(knapsack_constraints) or (epsilon > 0), "Specify non-zero epsilon for density search with knapsack constraints"
    assert ignorable_knapsack(knapsack_constraints) or (k > 0), "Specify k for density search with knapsack constraints"
    assert 0.0 <= epsilon <= 1.0, "Epsilon must be in [0, 1]"

    if opt_size_ub is None:
        opt_size_ub = len(gnd)

    num_sol, run_density_search, beta_scaling = init_sgs_params(num_sol, k, extendible, monotone, knapsack_constraints, epsilon)
    pq, num_fun, num_oracle = initialize_pq(gnd, f_diff, ind_add_oracle, num_sol, knapsack_constraints)

    if not pq:
        best_sol = None
        best_f_val = float('-inf')
        return best_sol, best_f_val, num_fun, num_oracle

    if run_density_search:
        delta = epsilon
        best_sol, best_f_val, num_f, num_or = density_search(pq, num_sol, beta_scaling, delta, f_diff, ind_add_oracle, knapsack_constraints, epsilon, opt_size_ub, verbose=verbose_lvl >= 2)
    else:
        density_ratio = 0.0
        best_sol, best_f_val, num_f, num_or, _ = simultaneous_greedy_alg(pq, num_sol, epsilon, f_diff, ind_add_oracle, knapsack_constraints, density_ratio, opt_size_ub, verbose=verbose_lvl >= 2)

    num_fun += num_f
    num_oracle += num_or

    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")

    # Đồng bộ GPU
    cp.cuda.Stream.null.synchronize()
    return best_sol, best_f_val, num_fun, num_oracle