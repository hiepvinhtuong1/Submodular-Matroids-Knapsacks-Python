import sys
import os

# Thêm thư mục gốc vào Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, project_root)

import time
import numpy as np
import networkx as nx
from typing import List, Set, Tuple, Union
from submodular_greedy import greedy, repeated_greedy, simultaneous_greedys, fantom, main_part_sprout
from itertools import combinations
import random


# Fix random seed
random.seed(1)
np.random.seed(1)

# Create an Erdos-Renyi graph
n = 1000  # Number of nodes
p = 0.01  # Probability of edge creation
erdos_graph = nx.erdos_renyi_graph(n, p, seed=1)

# Generate the weight matrix
weight_matrix = np.random.uniform(0, 1, (n, n))  # Equivalent to rand(Random.seed!(1), n, n)

# Ground set: all nodes
gnd = list(range(1, n + 1))

# Step 1: Construct objective function
def graph_cut(graph: nx.Graph, sol: Set[int]) -> float:
    sol = list(sol)
    total = 0.0
    for elm in sol:
        # Get all neighbors of elm (0-based in graph, but elm is 1-based)
        neighbors = list(graph.neighbors(elm - 1))
        # Filter neighbors not in sol (convert to 1-based for comparison)
        neighbors = [v + 1 for v in neighbors]  # Convert to 1-based
        arr_new = [v for v in neighbors if v not in sol]
        for e in arr_new:
            # Access weight_matrix (1-based indices)
            total += weight_matrix[elm - 1, e - 1]
    return total

def f_diff(elm: int, sol: Set[int]) -> float:
    return graph_cut(erdos_graph, set.union({elm}, sol)) - graph_cut(erdos_graph, sol)

# Step 2: Construct constraints
# Construct matroid constraints
card_limit = 10

def all_matroid_feasible(sol: Set[int], cardinality_limit: int) -> bool:
    # Cardinality constraint is feasible
    if len(sol) > cardinality_limit:
        return False
    return True

# Construct marginal independence oracle and independence oracle
ind_add_oracle = lambda elm, sol: all_matroid_feasible(set.union(sol, {elm}), card_limit)
ind_oracle = lambda sol: all_matroid_feasible(sol, card_limit)

budget_param = 1.0

# Construct knapsack constraints
knapsack_constraints = np.zeros((2, n))
# Knapsack 1: Degree
degree_array = np.array([erdos_graph.degree(v) for v in range(n)])  # 0-based indexing in graph
budget1 = 100 * budget_param
knapsack_constraints[0, :] = degree_array / budget1

# Knapsack 2: Modulo-based constraint
gnd_array = np.array(gnd)
budget2 = 40 * budget_param
knapsack_constraints[1, :] = (gnd_array % 10 + 1) / budget2

# Run algorithms
# 1: Greedy
sol, f_val, num_fun, num_oracle, knap_reject= greedy(gnd, f_diff, ind_add_oracle, knapsack_constraints=knapsack_constraints)
print("Greedy Solution:", sol, "Cut Value:", f_val)

# 2: Repeated Greedy
num_sol = 2
epsilon = 0.25
sol, f_val, num_fun, num_oracle = repeated_greedy(gnd, f_diff, ind_add_oracle, knapsack_constraints=knapsack_constraints, num_sol=num_sol, epsilon=epsilon, k=1)
print("Repeated Greedy Solution:", sol, "Cut Value:", f_val)

# 3: Simultaneous Greedy
sol, f_val, num_fun, num_oracle = simultaneous_greedys(gnd, f_diff, ind_add_oracle, knapsack_constraints=knapsack_constraints, extendible=True, num_sol=num_sol, epsilon=epsilon, k=1)
print("Simultaneous Greedy Solution:", sol, "Cut Value:", f_val)

# 4: FANTOM
sol, f_val, num_fun, num_oracle = fantom(gnd, f_diff, ind_add_oracle, knapsack_constraints=knapsack_constraints, epsilon=epsilon, k=1)
print("FANTOM Solution:", sol, "Cut Value:", f_val)

# 5: SPROUT++ and SPROUT
tc = 60  # tc = 60 as specified in Julia
param_c = 1
param_alpha = 0.5
mu = 1.0

def knapsack_feasible(sol: Set[int], knapsack_constraints: Union[np.ndarray, None]) -> bool:
    if knapsack_constraints is None:
        return True
    # Adjust indices: subtract 1 from each element in sol to convert from 1-based to 0-based
    adjusted_indices = [i - 1 for i in list(sol)]
    return np.all(np.sum(knapsack_constraints[:, adjusted_indices], axis=1) <= 1)

# SPROUT++ (converted from Julia)
def sproutpp(param_c: int, gnd: List[int], f_diff, ind_oracle, ind_add_oracle, num_sol: int = 0, k: int = 0, knapsack_constraints: Union[np.ndarray, None] = None, extendible: bool = False, monotone: bool = False, epsilon: float = 0.0, opt_size_ub: int = None, verbose_lvl: int = 1, param_alpha: float = 0.5) -> Tuple[Set[int], float, int]:
    # Start CPU time tracking (equivalent to CPUtic() in Julia)
    start_time = time.process_time()

    num_fun = 0
    best_sol = None
    best_f_val = 0

    # Generate all combinatorial sets
    gnd_combinatorials = list(combinations(gnd, param_c))
    random.shuffle(gnd_combinatorials)

    # Enumerate single-element sets
    fA_max = 0.0
    for elm in gnd:
        fA_max = max(fA_max, f_diff(elm, set()))
    num_fun += len(gnd)

    cnt = 0
    for gnd_combinatorial in gnd_combinatorials:
        gnd_combinatorial = set(gnd_combinatorial)
        # Check independence system and knapsack constraints
        if not ind_oracle(gnd_combinatorial) or not knapsack_feasible(gnd_combinatorial, knapsack_constraints):
            continue

        # Compute f(A)
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

        # Construct new objective function z(S) = f(S ∪ A) - f(A)
        def z_diff(elm: int, sol: Set[int]) -> float:
            return f_diff(elm, set.union(sol, gnd_combinatorial))

        num_fun += 1

        # Update the ground set N (CASE: SPROUT++)
        gnd_new = [x for x in gnd if x not in gnd_combinatorial]

        # Construct all matroid constraints
        card_limit_new = card_limit - param_c

        def ind_add_oracle_new(elm: int, sol: Set[int]) -> bool:
            return all_matroid_feasible(set.union(sol, {elm}), card_limit_new)

        # Decrease the capacity of all knapsack-cost functions
        knapsack_constraints_new = knapsack_constraints.copy() if knapsack_constraints is not None else None

        if knapsack_constraints_new is not None:
            # Knapsack constraint 1
            budget1_new = budget1 - sum(degree_array[i - 1] for i in gnd_combinatorial)  # Adjust for 0-based indexing
            knapsack_constraints_new[0, :] = degree_array / budget1_new

            # Knapsack constraint 2
            budget2_new = budget2 - sum(((i % 10) + 1) for i in gnd_combinatorial)
            knapsack_constraints_new[1, :] = (gnd_array % 10 + 1) / budget2_new

        # Run simultaneousGreedy (equivalent to main_part_sprout)
        sol, f_val, oracle_calls, _ = main_part_sprout(param_c, fA_value, gnd_new, z_diff, ind_add_oracle_new, knapsack_constraints=knapsack_constraints_new, extendible=extendible, num_sol=num_sol, epsilon=epsilon, k=k, mu=mu)
        num_fun += oracle_calls

        # Add set A at the end
        f_val += fA_value
        if sol is None:
            sol = set(gnd_combinatorial)
        else:
            sol = sol.union(gnd_combinatorial)

        if f_val > best_f_val:
            best_sol, best_f_val = sol, f_val

    # End CPU time tracking (equivalent to CPUtoc() in Julia)
    end_time = time.process_time()
    print(f"CPU time: {end_time - start_time} seconds")

    return best_sol, best_f_val, num_fun

# SPROUT (converted from Julia)
def sprout(param_c: int, gnd: List[int], f_diff, ind_oracle, ind_add_oracle, num_sol: int = 0, k: int = 0, knapsack_constraints: Union[np.ndarray, None] = None, extendible: bool = False, monotone: bool = False, epsilon: float = 0.0, opt_size_ub: int = None, verbose_lvl: int = 1, param_alpha: float = 0.5) -> Tuple[Set[int], float, int]:
    # Start CPU time tracking (equivalent to CPUtic() in Julia)
    start_time = time.process_time()

    num_fun = 0
    best_sol = None
    best_f_val = 0

    # Generate all combinatorial sets
    gnd_combinatorials = list(combinations(gnd, param_c))
    random.shuffle(gnd_combinatorials)

    # Enumerate single-element sets
    fA_max = 0.0
    for elm in gnd:
        fA_max = max(fA_max, f_diff(elm, set()))
    num_fun += len(gnd)

    print(f"--------------------------------fA_max: {fA_max}--------------------------------")

    cnt = 0
    for gnd_combinatorial in gnd_combinatorials:
        gnd_combinatorial = set(gnd_combinatorial)
        # Check independence system and knapsack constraints
        if not ind_oracle(gnd_combinatorial) or not knapsack_feasible(gnd_combinatorial, knapsack_constraints):
            continue

        # Compute f(A)
        fA_value = 0.0
        set_a = set()
        for elm in gnd_combinatorial:
            fA_value += f_diff(elm, set_a)
            num_fun += 1
            set_a.add(elm)

        # if fA_value < param_alpha * fA_max:  # Commented in Julia code
        #     continue

        print(f"--------------------------------{cnt} iteration--------------------------------")
        cnt += 1

        # Construct new objective function z(S) = f(S ∪ A) - f(A)
        def z_diff(elm: int, sol: Set[int]) -> float:
            return f_diff(elm, set.union(sol, gnd_combinatorial))

        num_fun += 1

        # Update the ground set N (CASE: SPROUT)
        gnd_new = [x for x in gnd if x not in gnd_combinatorial and f_diff(x, set(gnd_combinatorial)) <= (fA_value / param_c)]
        num_fun += n - len(gnd_combinatorial)

        # Construct all matroid constraints
        card_limit_new = card_limit - param_c

        def ind_add_oracle_new(elm: int, sol: Set[int]) -> bool:
            return all_matroid_feasible(set.union(sol, {elm}), card_limit_new)

        # Decrease the capacity of all knapsack-cost functions
        knapsack_constraints_new = knapsack_constraints.copy() if knapsack_constraints is not None else None

        if knapsack_constraints_new is not None:
            # Knapsack constraint 1
            budget1_new = budget1 - sum(degree_array[i - 1] for i in gnd_combinatorial)  # Adjust for 0-based indexing
            knapsack_constraints_new[0, :] = degree_array / budget1_new

            # Knapsack constraint 2
            budget2_new = budget2 - sum(((i % 10) + 1) for i in gnd_combinatorial)
            knapsack_constraints_new[1, :] = (gnd_array % 10 + 1) / budget2_new

        # Run simultaneousGreedy (equivalent to main_part_sprout)
        sol, f_val, oracle_calls, _ = main_part_sprout(param_c, fA_value, gnd_new, z_diff, ind_add_oracle_new, knapsack_constraints=knapsack_constraints_new, extendible=extendible, num_sol=num_sol, epsilon=epsilon, k=k, mu=mu)
        num_fun += oracle_calls

        # Add set A at the end
        f_val += fA_value
        if sol is None:
            sol = set(gnd_combinatorial)
        else:
            sol = sol.union(gnd_combinatorial)

        if f_val > best_f_val:
            best_sol, best_f_val = sol, f_val

        print("-------------------------")
        print(f"Best current value is: {best_f_val}")
        print(f"Best current solution set is: {best_sol}")
        print("-------------------------")

    # End CPU time tracking (equivalent to CPUtoc() in Julia)
    end_time = time.process_time()
    print(f"CPU time: {end_time - start_time} seconds")

    return best_sol, best_f_val, num_fun

# Run SPROUT++
print("\nRunning SPROUT++")
sol, f_val, _ = sproutpp(param_c, gnd, f_diff, ind_oracle, ind_add_oracle, knapsack_constraints=knapsack_constraints, extendible=True, num_sol=num_sol, epsilon=epsilon, k=1)
print(f"SPROUT++ Solution:", sol, "Cut Value:", f_val)

# Run SPROUT
print("\nRunning SPROUT")
sol, f_val, _ = sprout(param_c, gnd, f_diff, ind_oracle, ind_add_oracle, knapsack_constraints=knapsack_constraints, extendible=True, num_sol=num_sol, epsilon=epsilon, k=1)
print(f"SPROUT Solution:", sol, "Cut Value:", f_val)