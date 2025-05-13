import math
import numpy as np
from typing import List, Set, Tuple, Union
from submodular_greedy.utils.helper_funs import initialize_pq, printset, printlnset, dimension_check, ignorable_knapsack, num_knapsack

def deterministic_usm(gnd: List[int], f_diff, verbose: bool = False) -> Tuple[Set[int], float, int]:
    Y = set(gnd)
    X = set()
    f_val = 0
    num_fun = 0

    iter = 1
    for elm in gnd:
        a = f_diff(elm, X)
        Y_minus_elm = Y - {elm}
        b = -f_diff(elm, Y_minus_elm)
        num_fun += 2

        if verbose:
            print(f"\nIteration {iter} looking at element {elm}")
            print("Set X is ", end="")
            printlnset(X)
            print("Set Y is ", end="")
            printlnset(Y)
            print(f"The value a is {a} \nThe value b is {b}")

        if a >= b:
            X.add(elm)
            f_val += a
        else:
            Y.discard(elm)

        iter += 1

    return X, f_val, num_fun

def repeated_greedy_alg(pq: List, num_sol: int, f_diff, ind_add_oracle, knapsack_constraints: Union[np.ndarray, None], density_ratio: float, epsilon: float, opt_size_ub: int, verbose: bool = False) -> Tuple[Set[int], float, int, int, bool]:
    best_sol = None
    best_f_val = float('-inf')
    num_fun = 0
    num_oracle = 0
    knap_reject = False

    iter = 1
    for i in range(num_sol):
        from submodular_greedy.algorithms.greedy import greedy_with_pq
        sol, f_val, num_f, num_or, greedy_kr = greedy_with_pq(pq.copy(), f_diff, ind_add_oracle, knapsack_constraints=knapsack_constraints, density_ratio=density_ratio, epsilon=epsilon, opt_size_ub=opt_size_ub)

        knap_reject = knap_reject or greedy_kr

        if verbose:
            print(f"\nIteration {iter}")
            print("\tGreedy returned set ", end="")
            printset(sol)
            print(f" with value {f_val}")

        if f_val > best_f_val:
            best_sol = sol.copy()
            best_f_val = f_val

        num_fun += num_f
        num_oracle += num_or

        keys_to_remove = [k for _, k in pq if k[0] in sol]
        pq[:] = [(p, k) for p, k in pq if k[0] not in sol]

        sol, f_val, num_f = deterministic_usm(list(sol), f_diff)

        if f_val > best_f_val:
            best_sol = sol.copy()
            best_f_val = f_val

        num_fun += num_f

        if verbose:
            print("\tUnconstrained returned set ", end="")
            printset(sol)
            print(f" with value {f_val}")

        iter += 1

        if not pq:
            break

    return best_sol, best_f_val, num_fun, num_oracle, knap_reject

def repeated_greedy_density_search(pq: List, num_sol: int, f_diff, ind_add_oracle, knapsack_constraints: Union[np.ndarray, None], beta_scaling: float, delta: float, epsilon: float, opt_size_ub: int, verbose: bool = False) -> Tuple[Set[int], float, int, int]:
    # Sửa đoạn code lấy max_gain
    if not pq:  # Kiểm tra nếu pq rỗng
        return set(), 0.0, 0, 0

    neg_max_gain, _ = pq[0]  # Lấy phần tử đầu tiên của pq
    max_gain = -neg_max_gain  # Phủ định để có giá trị max_gain

    best_sol = None
    best_f_val = float('-inf')
    num_fun = 0
    num_oracle = 0

    lower_density_ratio = beta_scaling * max_gain * 1
    upper_density_ratio = beta_scaling * max_gain * opt_size_ub

    iter = 1
    while upper_density_ratio > (1 + delta) * lower_density_ratio:
        if verbose:
            print(f"\nIteration {iter}")
            print(f"\tUpper density ratio is {upper_density_ratio} and lower density ratio is {lower_density_ratio}")
            print(f"\tBest value seen is {best_f_val}")

        density_ratio = math.sqrt(lower_density_ratio * upper_density_ratio)

        sol, f_val, num_f, num_or, knap_reject = repeated_greedy_alg(pq.copy(), num_sol, f_diff, ind_add_oracle, knapsack_constraints, density_ratio, epsilon, opt_size_ub)

        if f_val > best_f_val:
            best_sol = sol.copy()
            best_f_val = f_val

        num_fun += num_f
        num_oracle += num_or

        if knap_reject:
            upper_density_ratio = density_ratio
        else:
            lower_density_ratio = density_ratio

        iter += 1

    return best_sol, best_f_val, num_fun, num_oracle

def init_rg_params(num_sol: int, k: int, monotone: bool, knapsack_constraints: Union[np.ndarray, None], epsilon: float) -> Tuple[int, bool, float]:
    usm_a = 3.0

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
        beta_scaling = 2 * (1 - epsilon)**2 / (k + 2 * m + 1 + usm_a * (num_sol - 1) / 2)
    else:
        if num_sol == 0:
            num_sol = int(math.floor(1 + math.sqrt(2 * (k + 2 * m + 1) / usm_a)))
        beta_scaling = 2 * (1 - epsilon) * (1 - 1 / num_sol - epsilon) / (k + 2 * m + 1 + usm_a * (num_sol - 1) / 2)

    return num_sol, run_density_search, beta_scaling

def repeated_greedy(gnd: List[int], f_diff, ind_add_oracle, num_sol: int = 0, k: int = 0, knapsack_constraints: Union[np.ndarray, None] = None, monotone: bool = False, epsilon: float = 0.0, opt_size_ub: int = None, verbose_lvl: int = 1) -> Tuple[Set[int], float, int, int]:
    assert (num_sol > 0) or (k > 0), "At least num_sol or k needs to be specified"
    assert dimension_check(gnd, knapsack_constraints), "More elements in knapsack constraints than in ground set"
    assert ignorable_knapsack(knapsack_constraints) or (epsilon > 0), "Specify non-zero epsilon for density search with knapsack constraints"
    assert ignorable_knapsack(knapsack_constraints) or (k > 0), "Specify k for density search with knapsack constraints"
    assert 0.0 <= epsilon <= 1.0, "Epsilon must be in [0, 1]"

    if opt_size_ub is None:
        opt_size_ub = max(gnd)

    num_sol, run_density_search, beta_scaling = init_rg_params(num_sol, k, monotone, knapsack_constraints, epsilon)
    pq, num_fun, num_oracle = initialize_pq(gnd, f_diff, ind_add_oracle, 1, knapsack_constraints)

    info_verbose = verbose_lvl >= 1
    alg_verbose = verbose_lvl >= 2

    if info_verbose:
        n = len(gnd)
        print("Running repeated greedys\n============================")
        print(f"Ground set has {n} elements")
        if k > 0:
            print(f"Independence system is {k}-system")
        else:
            print("The independence system parameter k is not specified")

        if (knapsack_constraints is not None) and ignorable_knapsack(knapsack_constraints):
            print("Knapsack constraints are always satisfied and thus will be ignored")
        elif not ignorable_knapsack(knapsack_constraints):
            m, n_k = knapsack_constraints.shape
            print(f"There are {m} knapsack constraints")

        print(f"\nConstructing {num_sol} solutions")
        if run_density_search:
            print("A grid search for the best density ratio will be run with the parameters:")
            print(f"\tbeta scaling term is {beta_scaling}")
            print(f"\terror term is {epsilon}")
            print(f"\tbound on largest set is {opt_size_ub}")

    if run_density_search:
        delta = epsilon
        best_sol, best_f_val, num_f, num_or = repeated_greedy_density_search(pq, num_sol, f_diff, ind_add_oracle, knapsack_constraints, beta_scaling, delta, 0.0, opt_size_ub, verbose=alg_verbose)
    else:
        density_ratio = 0.0
        best_sol, best_f_val, num_f, num_or, knap_reject = repeated_greedy_alg(pq, num_sol, f_diff, ind_add_oracle, knapsack_constraints, density_ratio, epsilon, opt_size_ub, verbose=alg_verbose)

    num_fun += num_f
    num_oracle += num_or

    if info_verbose:
        if len(best_sol) <= 10:
            print("\n\nObtained solution S = ", end="")
            printlnset(best_sol)
        print(f"Obtained solution has value {best_f_val}")
        print(f"Required {num_fun} function evaluations and {num_oracle} independence queries")

    return best_sol, best_f_val, num_fun, num_oracle