import time
from typing import List, Set, Tuple, Union
import cupy as cp

def knapsack_feasible(sol: Set[int], knapsack_constraints: Union[cp.ndarray, None]) -> bool:
    if knapsack_constraints is None:
        return True
    sol_list = cp.array(list(sol), dtype=cp.int64) - 1
    if not sol_list.size:
        return True
    return cp.all(cp.sum(knapsack_constraints[:, sol_list], axis=1) <= 1).item()

def fantom(gnd: List[int], f_diff, ind_add_oracle, knapsack_constraints: Union[cp.ndarray, None] = None, epsilon: float = 0.5, k: int = 0) -> Tuple[Set[int], float, int, int]:
    start_time = time.time()
    num_fun = 0
    num_oracle = 0

    n = len(gnd)
    # Vector hóa tính vals_elements
    gnd_array = cp.array(gnd, dtype=cp.int64)
    # Giả sử f_diff_vectorized đã được định nghĩa trong two_knapsacks.py để tính marginal gain cho nhiều phần tử
    try:
        from two_knapsacks import f_diff_vectorized
        vals_elements = f_diff_vectorized(gnd)
    except ImportError:
        # Nếu không có f_diff_vectorized, dùng vòng lặp
        vals_elements = cp.array([f_diff(elm, set()) for elm in gnd])
    num_fun += len(gnd)
    M = cp.max(vals_elements).item()
    omega = gnd.copy()

    p = k
    ell = 0 if knapsack_constraints is None else knapsack_constraints.shape[0]
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
        sols, num_f, num_oracle_batch = iterated_gdt(p, ell, f_diff, ind_add_oracle, ground=omega, knapsack_constraints=knapsack_constraints, rho=rho)
        for sol in sols:
            U.append(sol)
            current_val = 0.0
            set_a = set()
            for elm in sol:
                current_val += f_diff(elm, set_a)
                num_f += 1
                set_a.add(elm)
        num_fun += num_f
        num_oracle += num_oracle_batch

    best_sol = set()
    best_f_val = 0
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
    print(f"Time taken: {end_time - start_time} seconds")
    # Đồng bộ GPU trước khi trả về
    cp.cuda.Stream.null.synchronize()
    return best_sol, best_f_val, num_fun, num_oracle

def iterated_gdt(p: int, ell: int, f_diff, ind_add_oracle, ground: List[int], knapsack_constraints: Union[cp.ndarray, None] = None, rho: float = 0.0) -> Tuple[List[Set[int]], int, int]:
    num_fun = 0
    num_oracle = 0

    S_i = []
    ground = ground.copy()
    for i in range(1, p + 2):
        S, nf, noc = gdt(p, ell, f_diff, ind_add_oracle, ground=ground, knapsack_constraints=knapsack_constraints, rho=rho)
        num_fun += nf
        num_oracle += noc
        S_i.append(S.copy())
        ground = [x for x in ground if x not in S]

    return S_i, num_fun, num_oracle

def gdt(p: int, ell: int, f_diff, ind_add_oracle, ground: List[int], knapsack_constraints: Union[cp.ndarray, None] = None, rho: float = 0.0) -> Tuple[Set[int], int, int]:
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
            union_set = S | {elm}
            if ind_add_oracle(elm, S) and knapsack_feasible(union_set, knapsack_constraints):
                val = f_diff(elm, S)
                num_fun += 1
                cost = cp.sum(knapsack_constraints[:, elm-1]).item() if knapsack_constraints is not None else 0
                if val / (cost + 1e-6) >= rho:
                    if val > cand_val:
                        cand = elm
                        cand_val = val
                        flag = True
        if cand != -1:
            S.add(cand)

    return S, num_fun, num_oracle