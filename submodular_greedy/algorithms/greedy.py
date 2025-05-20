import random
import cupy as cp
from typing import List, Set, Tuple, Union
from submodular_greedy.utils.helper_funs import initialize_pq, printset
from submodular_greedy.algorithms.simultaneous_greedys import simultaneous_greedy_alg

def greedy_with_pq(pq: List, f_diff, ind_add_oracle, knapsack_constraints: Union[cp.ndarray, None] = None, density_ratio: float = 0.0, epsilon: float = 0.0, opt_size_ub: int = None, verbose: bool = False) -> Tuple[Set[int], float, int, int, bool]:
    if opt_size_ub is None:
        opt_size_ub = len(pq)
    best_sol, best_f_val, num_fun, num_oracle, knap_reject = simultaneous_greedy_alg(pq, 1, epsilon, f_diff, ind_add_oracle, knapsack_constraints, density_ratio, opt_size_ub, verbose=verbose)
    # Đồng bộ GPU
    cp.cuda.Stream.null.synchronize()
    return best_sol, best_f_val, num_fun, num_oracle, knap_reject

def greedy(gnd: List[int], f_diff, ind_add_oracle, knapsack_constraints: Union[cp.ndarray, None] = None, density_ratio: float = 0.0, epsilon: float = 0.0, opt_size_ub: int = None, verbose: bool = False) -> Tuple[Set[int], float, int, int, bool]:
    if opt_size_ub is None:
        opt_size_ub = len(gnd)

    pq, num_fun, num_oracle = initialize_pq(gnd, f_diff, ind_add_oracle, 1, knapsack_constraints)
    best_sol, best_f_val, num_f, num_or, knap_reject = greedy_with_pq(pq, f_diff, ind_add_oracle, knapsack_constraints=knapsack_constraints, density_ratio=density_ratio, epsilon=epsilon, opt_size_ub=opt_size_ub, verbose=verbose)

    num_fun += num_f
    num_oracle += num_or

    # Đồng bộ GPU
    cp.cuda.Stream.null.synchronize()
    return best_sol, best_f_val, num_fun, num_oracle, knap_reject

def sample_greedy(gnd: List[int], sample_prob: float, f_diff, ind_add_oracle, verbose: bool = False) -> Tuple[Set[int], float, int, int]:
    assert 0 <= sample_prob <= 1, "Sample probability must be between 0 and 1"

    sample_gnd = [elm for elm in gnd if random.random() <= sample_prob]

    if verbose:
        print("The subsampled elements are ", sample_gnd)

    best_sol, best_f_val, num_fun, num_oracle, _ = greedy(sample_gnd, f_diff, ind_add_oracle, knapsack_constraints=None, density_ratio=0.0, verbose=verbose)
    # Đồng bộ GPU
    cp.cuda.Stream.null.synchronize()
    return best_sol, best_f_val, num_fun, num_oracle