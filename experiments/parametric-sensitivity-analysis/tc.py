import random
import numpy as np
import time
import networkx as nx
from typing import Set, List, Union, Tuple
from itertools import combinations
from submodular_greedy import greedy, repeated_greedy, simultaneous_greedys, fantom, main_part_sprout

n = 1000
erdos_graph = nx.erdos_renyi_graph(n, 0.01, seed=1)

# Tạo ma trận trọng số ngẫu nhiên
random.seed(1)
weight_matrix = np.random.rand(n, n)

# Bước 1: Xây dựng hàm mục tiêu
def graph_cut(graph: nx.Graph, sol: Set[int]) -> float:
    sol = list(sol)
    total = 0.0
    for elm in sol:
        neighbors = list(graph.neighbors(elm))
        neighbors_not_in_sol = [x for x in neighbors if x not in sol]
        for e in neighbors_not_in_sol:
            total += weight_matrix[elm, e]
    return total

def f_diff(elm: int, sol: Set[int]) -> float:
    return graph_cut(erdos_graph, set.union({elm}, sol)) - graph_cut(erdos_graph, sol)

# Bước 2: Xây dựng các ràng buộc
# Ràng buộc matroid
card_limit = 10

def all_matroid_feasible(sol: Set[int], cardinality_limit: int) -> bool:
    if len(sol) > cardinality_limit:
        return False
    return True

def ind_add_oracle(elm: int, sol: Set[int]) -> bool:
    return all_matroid_feasible(set.union(sol, {elm}), card_limit)

def ind_oracle(sol: Set[int]) -> bool:
    return all_matroid_feasible(sol, card_limit)

# Ràng buộc knapsack
budget_param = 1.0
knapsack_constraints = np.zeros((2, n))
budget1 = 100 * budget_param
degree_array = np.array([erdos_graph.degree(i) for i in range(n)])
knapsack_constraints[0, :] = degree_array / budget1

gnd = list(range(n))
budget2 = 40 * budget_param
knapsack_constraints[1, :] = (np.array(gnd) % 10 + 1) / budget2

# Chạy các thuật toán
# 1: Greedy
sol, f_val, _, _, _ = greedy(gnd, f_diff, ind_add_oracle, knapsack_constraints=knapsack_constraints)
print("Greedy:", sol, f_val)

# 2: Repeated Greedy
num_sol = 2
epsilon = 0.25
sol, f_val, _, _ = repeated_greedy(gnd, f_diff, ind_add_oracle, knapsack_constraints=knapsack_constraints,
                                    num_sol=num_sol, epsilon=epsilon, k=1)
print("Repeated Greedy:", sol, f_val)

# 3: Simultaneous Greedy
sol, f_val, _, _ = simultaneous_greedys(gnd, f_diff, ind_add_oracle, knapsack_constraints=knapsack_constraints,
                                        extendible=True, num_sol=num_sol, epsilon=epsilon, k=1)
print("Simultaneous Greedy:", sol, f_val)

# 4: FANTOM
sol, f_val, _, _ = fantom(gnd, f_diff, ind_add_oracle, knapsack_constraints=knapsack_constraints,
                          epsilon=epsilon, k=1)
print("FANTOM:", sol, f_val)

# 5: SPROUT++
param_c = 1
ratio_tc_to_n = 0.06
tc = int(n * ratio_tc_to_n)
mu = 1.0

def knapsack_feasible(sol: Set[int], knapsack_constraints: Union[np.ndarray, None]) -> bool:
    if knapsack_constraints is None:
        return True
    else:
        sol = list(sol)
        return np.all(np.sum(knapsack_constraints[:, sol], axis=1) <= 1)

def sproutpp(param_c: int, gnd: List[int], f_diff, ind_oracle, ind_add_oracle, num_sol: int = 0, k: int = 0,
             knapsack_constraints: Union[np.ndarray, None] = None, extendible: bool = False,
             monotone: bool = False, epsilon: float = 0.0, opt_size_ub: int = None,
             verbose_lvl: int = 1, param_alpha: float = 0.5) -> Tuple[Set[int], float, int]:
    start_time = time.time()
    num_fun = 0
    best_sol = None
    best_f_val = 0.0

    if opt_size_ub is None:
        opt_size_ub = len(gnd)

    gnd_combinatorials = list(combinations(gnd, param_c))
    random.shuffle(gnd_combinatorials)

    fA_max = 0.0
    for elm in gnd:
        fA_max = max(fA_max, f_diff(elm, set()))
    num_fun += len(gnd)

    cnt = 0
    for gnd_combinatorial in gnd_combinatorials:
        gnd_combinatorial = set(gnd_combinatorial)
        if ind_oracle(gnd_combinatorial) == False or knapsack_feasible(gnd_combinatorial, knapsack_constraints) == False:
            continue

        fA_value = 0.0
        set_a = set()
        for elm in gnd_combinatorial:
            fA_value += f_diff(elm, set_a)
            num_fun += 1
            set_a.add(elm)

        if fA_value < param_alpha * fA_max:
            continue

        if cnt >= tc:
            break
        cnt += 1

        def z_diff(elm: int, sol: Set[int]) -> float:
            return f_diff(elm, sol.union(gnd_combinatorial))
        num_fun += 1

        gnd_new = [x for x in gnd if x not in gnd_combinatorial]
        card_limit_new = card_limit - param_c

        def ind_add_oracle_new(elm: int, sol: Set[int]) -> bool:
            return all_matroid_feasible(set.union(sol, {elm}), card_limit_new)

        knapsack_constraints_new = knapsack_constraints.copy()
        budget1_new = budget1 - np.sum(degree_array[list(gnd_combinatorial)])
        knapsack_constraints_new[0, :] = degree_array / budget1_new

        budget2_new = budget2 - np.sum((np.array(list(gnd_combinatorial)) % 10 + 1))
        knapsack_constraints_new[1, :] = (np.array(gnd) % 10 + 1) / budget2_new

        sol, f_val, oracle_calls, _ = main_part_sprout(
            param_c, fA_value, gnd_new, z_diff, ind_add_oracle_new, knapsack_constraints=knapsack_constraints_new,
            extendible=True, num_sol=num_sol, epsilon=epsilon, k=1, mu=mu)
        num_fun += oracle_calls

        f_val += fA_value
        if sol is None:
            sol = gnd_combinatorial
        else:
            sol = sol.union(gnd_combinatorial)

        if f_val > best_f_val:
            best_sol, best_f_val = sol, f_val

        print("-------------------------")
        print(f"Best current value is: {best_f_val}")
        print(f"Best current solution set is: {best_sol}")
        print("-------------------------")

    end_time = time.time()
    print(f"SPROUT++ runtime: {end_time - start_time} seconds")
    return best_sol, best_f_val, num_fun

sol, f_val, _ = sproutpp(param_c, gnd, f_diff, ind_oracle, ind_add_oracle, knapsack_constraints=knapsack_constraints,
                         extendible=True, num_sol=num_sol, epsilon=epsilon, k=1, param_alpha=0.5)
print("SPROUT++:", sol, f_val)