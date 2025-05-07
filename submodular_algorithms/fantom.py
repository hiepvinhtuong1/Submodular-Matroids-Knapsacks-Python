from typing import Set, List, Union, Tuple
import numpy as np
import time
from ..utils.helper_funs import num_knapsack

def knapsack_feasible(sol: Set[int], knapsack_constraints: Union[np.ndarray, None]) -> bool:
    if knapsack_constraints is None:
        return True
    else:
        sol = list(sol)
        return np.all(np.sum(knapsack_constraints[:, sol], axis=1) <= 1)

def GDT(p: int, ell: int, f_diff, ind_add_oracle, ground: List[int], knapsack_constraints: Union[np.ndarray, None], rho: float) -> Tuple[Set[int], int, int]:
    num_fun = 0
    num_oracle = 0
    S = set()
    flag = True

    while flag:
        flag = False
        cand = -1
        cand_val = -1

        for elm in ground:
            num_oracle += 1
            if ind_add_oracle(elm, S) and knapsack_feasible(S.union({elm}), knapsack_constraints):
                val = f_diff(elm, S)
                num_fun += 1
                if knapsack_constraints is None:
                    density = 0.0
                else:
                    density = np.sum(knapsack_constraints[:, elm])
                if val / (density + 1e-6) >= rho:
                    if val > cand_val:
                        cand = elm
                        cand_val = val
                        flag = True

        if cand != -1:
            S.add(cand)

    return S, num_fun, num_oracle

def iterated_GDT(p: int, ell: int, f_diff, ind_add_oracle, ground: List[int], knapsack_constraints: Union[np.ndarray, None], rho: float) -> Tuple[List[Set[int]], int, int]:
    num_fun = 0
    num_oracle = 0
    print(f"--------------rho: {rho}---------------")

    S_i = []
    ground_copy = ground.copy()

    for i in range(1, p + 2):
        S, nf, noc = GDT(p, ell, f_diff, ind_add_oracle, ground=ground_copy, knapsack_constraints=knapsack_constraints, rho=rho)
        num_fun += nf
        num_oracle += noc
        S_i.append(S.copy())
        ground_copy = [x for x in ground_copy if x not in S]

    return S_i, num_fun, num_oracle

def fantom(gnd: List[int], f_diff, ind_add_oracle, knapsack_constraints: Union[np.ndarray, None] = None, epsilon: float = 0.5, k: int = 0) -> Tuple[Set[int], float, int, int]:
    start_time = time.time()
    num_fun = 0
    num_oracle = 0

    n = len(gnd)
    vals_elements = [f_diff(elm, set()) for elm in gnd]
    num_fun += len(gnd)
    M = max(vals_elements)
    omega = gnd.copy()

    p = k
    ell = num_knapsack(knapsack_constraints)
    gamma = (2 * p * M) / ((p + 1) * (2 * p + 1))
    R = []
    val = 1

    while val <= n:
        R.append(val * gamma)
        val *= (1 + epsilon)

    len_R = len(R)
    print(f"--------------len_R: {len_R}---------------")

    U = []
    for rho in R:
        omega = gnd.copy()
        sols, num_f, num_oracle_i = iterated_GDT(p, ell, f_diff, ind_add_oracle, ground=omega, knapsack_constraints=knapsack_constraints, rho=rho)
        for sol in sols:
            U.append(sol)
            current_val = 0.0
            set_a = set()
            for elm in sol:
                current_val += f_diff(elm, set_a)
                num_f += 1
                set_a.add(elm)
        num_fun += num_f
        num_oracle += num_oracle_i

    best_sol = []
    best_f_val = 0.0

    for sol in U:
        current_val = 0.0
        set_a = set()
        for elm in sol:
            current_val += f_diff(elm, set_a)
            num_fun += 1
            set_a.add(elm)

        if current_val > best_f_val:
            best_f_val = current_val
            best_sol = sol.copy()

    end_time = time.time()
    print(f"FANTOM runtime: {end_time - start_time} seconds")

    return best_sol, best_f_val, num_fun, num_oracle