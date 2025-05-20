from typing import List, Set, Tuple
import cupy as cp

from submodular_greedy.algorithms import knapsack_feasible

def algorithm3_itw(gnd: List[int], f_diff, ind_add_oracle, ind_oracle, knapsack_constraints: cp.ndarray, budget1: float,
                   budget2: float, rating_array: cp.ndarray, date_array: cp.ndarray, max_rating: float, year1: int,
                   mu: float = 1.0) -> Tuple[Set[int], float, int, int]:
    """
    Algorithm3ITWAlgorithm for submodular maximization under matroid and knapsack constraints.

    Args:
        gnd: List of elements in the ground set.
        f_diff: Function to compute the marginal gain f(e | S).
        ind_add_oracle: Function to check if adding an element to a solution is feasible.
        ind_oracle: Function to check if a solution is feasible.
        knapsack_constraints: Array of knapsack constraints (CuPy array).
        budget1, budget2: Budgets for knapsack constraints.
        rating_array, date_array: Arrays for ratings and years (CuPy arrays).
        max_rating, year1: Parameters for cost computation.
        mu: Smoothing parameter for marginal gains.

    Returns:
        Tuple containing the solution set, objective value, number of function evaluations, and number of oracle queries.
    """
    num_fun = 0
    num_oracle = 0

    # Step 1: Split the ground set into E1 and E2
    gnd_array = cp.array(gnd, dtype=cp.int64)
    idx_array = gnd_array - 1  # Chỉ số 0-based
    c1 = max_rating - rating_array[idx_array]  # Cost for rating constraint
    c2 = cp.abs(year1 - date_array[idx_array])  # Cost for year constraint
    num_oracle += 2 * len(gnd)  # Đếm số truy vấn oracle cho toàn bộ ground set

    # Vector hóa điều kiện để chia E1 và E2
    condition = (c1 > budget1 / 2) | (c2 > budget2 / 2)
    E1 = cp.where(condition)[0].get() + 1  # Chuyển về 1-based và lấy list
    E2 = cp.where(~condition)[0].get() + 1
    E1 = list(E1)
    E2 = list(E2)

    # Step 2: Find the best element in E1
    best_em = None
    best_em_val = float('-inf')
    if E1:
        for e in E1:
            if not ind_oracle({e}) or not knapsack_feasible({e}, knapsack_constraints):
                num_oracle += 2
                continue
            val = f_diff(e, set())
            num_fun += 1
            num_oracle += 1
            if val > best_em_val:
                best_em = e
                best_em_val = val

    # Step 3: Build S1 and S2 from E2 with smoothing
    S1 = set()
    S2 = set()
    elements_added_S1 = []
    elements_added_S2 = []

    remaining = set(E2)
    while remaining:
        best_e = None
        best_i = None
        best_gain = float('-inf')

        # Find feasible elements for S1 and S2
        M1 = set()
        M2 = set()
        for e in remaining:
            if ind_add_oracle(e, S1) and knapsack_feasible(S1 | {e}, knapsack_constraints):
                M1.add(e)
            if ind_add_oracle(e, S2) and knapsack_feasible(S2 | {e}, knapsack_constraints):
                M2.add(e)
            num_oracle += 2

        if not M1 and not M2:
            break

        # Compute smoothed marginal gains with weighting
        for e in M1:
            gain = f_diff(e, S1)
            num_fun += 1
            smoothed_gain = mu * gain + (1 - mu) * (1 / (len(S1) + 1))  # Smoothing with size-based weighting
            if smoothed_gain > best_gain:
                best_gain = smoothed_gain
                best_e = e
                best_i = 1

        for e in M2:
            gain = f_diff(e, S2)
            num_fun += 1
            smoothed_gain = mu * gain + (1 - mu) * (1 / (len(S2) + 1))
            if smoothed_gain > best_gain:
                best_gain = smoothed_gain
                best_e = e
                best_i = 2

        if best_gain <= 0 or best_e is None:
            break

        if best_i == 1:
            S1.add(best_e)
            elements_added_S1.append(best_e)
        else:
            S2.add(best_e)
            elements_added_S2.append(best_e)

        remaining.remove(best_e)

    # Step 4: Generate solution sets from S1 and S2
    candidates = []
    if best_em is not None:
        candidates.append(({best_em}, best_em_val))

    # Generate X' and Y' from S1 and S2
    for t in range(1, len(elements_added_S1) + 1):
        X_prime = set(elements_added_S1[-t:])
        if ind_oracle(X_prime) and knapsack_feasible(X_prime, knapsack_constraints):
            val = sum(f_diff(e, set(elements_added_S1[:i]) - {e}) for i, e in enumerate(elements_added_S1[-t:]))
            num_fun += t
            num_oracle += 1
            candidates.append((X_prime, val))

    for t in range(1, len(elements_added_S2) + 1):
        Y_prime = set(elements_added_S2[-t:])
        if ind_oracle(Y_prime) and knapsack_feasible(Y_prime, knapsack_constraints):
            val = sum(f_diff(e, set(elements_added_S2[:i]) - {e}) for i, e in enumerate(elements_added_S2[-t:]))
            num_fun += t
            num_oracle += 1
            candidates.append((Y_prime, val))

    # Step 5: Select the best solution
    best_sol = set()
    best_f_val = float('-inf')
    for sol, val in candidates:
        if val > best_f_val:
            best_sol = sol
            best_f_val = val

    # Đồng bộ GPU trước khi trả về kết quả
    cp.cuda.Stream.null.synchronize()
    return best_sol, best_f_val, num_fun, num_oracle