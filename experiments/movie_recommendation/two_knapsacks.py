import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, project_root)
import time
import pandas as pd
import cupy as cp
from typing import List, Set, Union, Tuple
from itertools import combinations
import random
import ast
from submodular_greedy import greedy, repeated_greedy, simultaneous_greedys, fantom, main_part_sprout, twingreedy2,algorithm3_itw
from submodular_greedy.algorithms.fantom import knapsack_feasible



# Đọc dữ liệu
data_file = os.path.join(project_root, "data", "movie_info100.csv")
movie_info_df = pd.read_csv(data_file)

n = len(movie_info_df)
print(f"Number of movies: {n}")

# Chuyển dữ liệu sang mảng CuPy
movie_info_df['vec'] = movie_info_df['vec'].apply(ast.literal_eval)
vec_list = movie_info_df['vec'].tolist()
rating_array = cp.array(movie_info_df['rating'].to_numpy(), dtype=cp.float32)
date_array = cp.array(movie_info_df['year'].to_numpy(), dtype=cp.float32)
max_rating = 10
year1 = 1995
budget_param = 1
budget1 = 20 * budget_param
budget2 = 30 * budget_param

# Khởi tạo knapsack constraints
knapsack_constraints = cp.zeros((2, n), dtype=cp.float32)
knapsack_constraints[0, :] = (max_rating - rating_array) / budget1
knapsack_constraints[1, :] = cp.abs(year1 - date_array) / budget2


# Hàm tính similarity array
def compute_similarity_array(vec_list: List[List[float]], sigma: float = 1.0) -> cp.ndarray:
    vec_array = cp.array(vec_list, dtype=cp.float32)
    n = len(vec_list)
    i, j = cp.triu_indices(n, k=0)
    distances = cp.sqrt(cp.sum((vec_array[i] - vec_array[j]) ** 2, axis=1))
    similarity_array = cp.exp(-4 * distances)
    return similarity_array


similarity_array = compute_similarity_array(vec_list)
print(f"Similarity array size: {similarity_array.size} elements")


# Hàm mục tiêu
def ij_to_ind(i: int, j: int, n: int) -> int:
    if i > j:
        i, j = j, i
    return (i - 1) * n - (i * (i - 1)) // 2 + j - 1


def get_similarity(similarity_array: cp.ndarray, i: int, j: int, n: int) -> float:
    ind = ij_to_ind(i, j, n)
    return float(similarity_array[ind])


def dispersion_diff_vectorized(elms: cp.ndarray, sol: Set[int], similarity_array: cp.ndarray, n: int) -> cp.ndarray:
    results = cp.zeros(len(elms), dtype=cp.float32)
    sol_array = cp.array(list(sol), dtype=cp.int64) if sol else cp.array([], dtype=cp.int64)
    indices = cp.arange(1, n + 1)

    for idx, elm in enumerate(elms):
        if elm in sol:
            continue
        coverage_term = cp.sum(cp.array([get_similarity(similarity_array, int(i), int(elm), n) for i in indices]))
        diversity_term = 2 * cp.sum(
            cp.array([get_similarity(similarity_array, int(i), int(elm), n) for i in sol_array])) + \
                         get_similarity(similarity_array, int(elm), int(elm), n)
        results[idx] = (coverage_term - diversity_term) / n
    return results


def f_diff(elm: int, sol: Set[int]) -> float:
    elm_array = cp.array([elm], dtype=cp.int64)
    return float(dispersion_diff_vectorized(elm_array, sol, similarity_array, n)[0])


def f_diff_vectorized(elms: List[int]) -> cp.ndarray:
    elm_array = cp.array(elms, dtype=cp.int64)
    return dispersion_diff_vectorized(elm_array, set(), similarity_array, n)


# Ràng buộc matroid
movie_info_df['genre_set'] = movie_info_df['genre_set'].apply(
    lambda x: set(ast.literal_eval(x.replace('Set([', '[').replace('])', ']'))))
movie_genre_df = movie_info_df['genre_set'].tolist()
genre_list = sorted(set().union(*movie_genre_df))
card_limit = 10
genre_card_limit = 2
genre_limit = {genre: genre_card_limit for genre in genre_list}


def all_matroid_feasible(sol: Set[int], cardinality_limit: int, genre_limit: dict, genre_list: List[str],
                         movie_genre_df: List[Set[str]]) -> bool:
    if len(sol) > cardinality_limit:
        return False
    sol_array = cp.array(list(sol), dtype=cp.int64) - 1
    genre_count = cp.zeros(len(genre_list), dtype=cp.int64)
    for idx in sol_array:
        for genre in movie_genre_df[int(idx)]:
            if genre in genre_list:
                genre_idx = genre_list.index(genre)
                genre_count[genre_idx] += 1
    return cp.all(genre_count <= cp.array([genre_limit[genre] for genre in genre_list])).item()


ind_add_oracle = lambda elm, sol: all_matroid_feasible(sol | {elm}, card_limit, genre_limit, genre_list, movie_genre_df)
ind_oracle = lambda sol: all_matroid_feasible(sol, card_limit, genre_limit, genre_list, movie_genre_df)


# Hàm knapsack_feasible
def knapsack_feasible(sol: Set[int], knapsack_constraints: cp.ndarray) -> bool:
    if knapsack_constraints is None:
        return True
    sol_list = cp.array(list(sol), dtype=cp.int64) - 1
    if not sol_list.size:
        return True
    costs = cp.sum(knapsack_constraints[:, sol_list], axis=1)
    return cp.all(costs <= 1).item()


# Hàm SPROUT++
tc = 2000
param_c = 1
param_alpha = 0.5
mu = 1.0


def sproutpp(param_c: int, gnd: List[int], f_diff, ind_oracle, ind_add_oracle, num_sol: int = 0, k: int = 0,
             knapsack_constraints: Union[cp.ndarray, None] = None, extendible: bool = False, monotone: bool = False,
             epsilon: float = 0.0, opt_size_ub: int = None, verbose_lvl: int = 1, param_alpha: float = 0.5) -> Tuple[
    Set[int], float, int]:
    num_fun = 0
    best_sol = None
    best_f_val = 0
    gnd_array = cp.array(gnd, dtype=cp.int64)
    gnd_combinatorials = list(combinations(gnd, param_c))
    random.shuffle(gnd_combinatorials)

    fA_max = cp.max(f_diff_vectorized(gnd)).item()
    num_fun += len(gnd)
    cnt = 0
    batch_size = 128  # Giảm batch_size để quản lý bộ nhớ

    for batch_idx in range(0, min(len(gnd_combinatorials), tc * batch_size), batch_size):
        batch = gnd_combinatorials[batch_idx:batch_idx + batch_size]
        batch_array = cp.array(batch, dtype=cp.int64)

        feasible = cp.ones(len(batch), dtype=cp.bool)
        for i in range(len(batch)):
            comb_set = set(batch[i].get())
            feasible[i] = ind_oracle(comb_set) and knapsack_feasible(comb_set, knapsack_constraints)

        fA_values = cp.zeros(len(batch), dtype=cp.float32)
        for i in range(len(batch)):
            if not feasible[i]:
                continue
            comb_set = set(batch[i].get())
            set_a = set()
            fA_value = 0.0
            for elm in comb_set:
                fA_value += f_diff(elm, set_a)
                set_a.add(elm)
                num_fun += 1
            fA_values[i] = fA_value

        valid_mask = (fA_values >= param_alpha * fA_max) & feasible
        valid_indices = cp.where(valid_mask)[0]

        for i in valid_indices:
            if cnt >= tc:
                break
            cnt += 1
            print(f"-------------------------------- {cnt} iteration --------------------------------")
            gnd_combinatorial = set(batch[i].get())
            fA_value = float(fA_values[i])

            def z_diff(elm: int, sol: Set[int]) -> float:
                return f_diff(elm, sol | gnd_combinatorial)

            num_fun += 1
            gnd_new = [x for x in gnd if x not in gnd_combinatorial]
            card_limit_new = card_limit - param_c
            genre_limit_new = genre_limit.copy()
            for elm in gnd_combinatorial:
                for genre in movie_genre_df[elm - 1]:
                    if genre in genre_list:
                        genre_limit_new[genre] = max(genre_limit_new[genre] - 1, 0)

            def ind_add_oracle_new(elm: int, sol: Set[int]) -> bool:
                return all_matroid_feasible(sol | {elm}, card_limit_new, genre_limit_new, genre_list, movie_genre_df)

            knapsack_constraints_new = knapsack_constraints.copy()
            adjusted_indices = cp.array([i - 1 for i in gnd_combinatorial], dtype=cp.int64)
            budget1_new = budget1 - cp.sum(max_rating - rating_array[adjusted_indices]).item()
            budget2_new = budget2 - cp.sum(cp.abs(year1 - date_array[adjusted_indices])).item()
            if budget1_new <= 0 or budget2_new <= 0:
                continue
            knapsack_constraints_new[0, :] = (max_rating - rating_array) / budget1_new
            knapsack_constraints_new[1, :] = cp.abs(year1 - date_array) / budget2_new

            sol, f_val, oracle_calls, _ = main_part_sprout(param_c, fA_value, gnd_new, z_diff, ind_add_oracle_new,
                                                           knapsack_constraints=knapsack_constraints_new,
                                                           extendible=True,
                                                           num_sol=num_sol, epsilon=epsilon, k=20, mu=mu)
            num_fun += oracle_calls
            f_val += fA_value
            if sol is None:
                sol = gnd_combinatorial.copy()
            else:
                sol |= gnd_combinatorial
            if f_val > best_f_val:
                best_sol, best_f_val = sol.copy(), f_val

    return best_sol, best_f_val, num_fun


# Hàm chạy thuật toán
def run_algorithm(name, func, *args, **kwargs):
    cp.cuda.Stream.null.synchronize()
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()

    if name == "SPROUT++":
        sol, f_val, num_fun = result
        num_oracle = None
    else:
        sol, f_val, num_fun, num_oracle = result[:4]

    running_time = end_time - start_time
    result_dict = {
        "Algorithm": name,
        "budget": budget_param,
        "f(S)": f_val,
        "Number of query": num_fun,
        "running time": running_time
    }

    output_file = os.path.join(project_root, "data", "algorithm_results.csv")
    result_df = pd.DataFrame([result_dict])
    if not os.path.exists(output_file):
        result_df.to_csv(output_file, index=False, mode='w')
    else:
        result_df.to_csv(output_file, index=False, mode='a', header=False)

    print(f"Saved result for {name} to {output_file}")
    print(f"Result for {name}:")
    print(result_df)
    print(f"GPU memory allocated: {cp.get_default_memory_pool().used_bytes() / 1024 ** 2:.2f} MB")

    return result_dict


# Chạy các thuật toán
gnd = list(range(1, n + 1))
num_sol = 2
epsilon = 0.25

print("Running Greedy...")
run_algorithm("Greedy", greedy, gnd, f_diff, ind_add_oracle, knapsack_constraints=knapsack_constraints)

print("Running Repeated Greedy...")
run_algorithm("Repeated Greedy", repeated_greedy, gnd, f_diff, ind_add_oracle,
              knapsack_constraints=knapsack_constraints, num_sol=num_sol, epsilon=epsilon, k=20)

print("Running Simultaneous Greedy...")
run_algorithm("Simultaneous Greedy", simultaneous_greedys, gnd, f_diff, ind_add_oracle,
              knapsack_constraints=knapsack_constraints, extendible=True, num_sol=num_sol,
              epsilon=epsilon, k=20)

print("Running FANTOM...")
run_algorithm("FANTOM", fantom, gnd, f_diff, ind_add_oracle, knapsack_constraints=knapsack_constraints,
              epsilon=epsilon, k=20)

print("Running SPROUT++...")
run_algorithm("SPROUT++", sproutpp, param_c, gnd, f_diff, ind_oracle, ind_add_oracle,
              knapsack_constraints=knapsack_constraints, extendible=True, num_sol=num_sol,
              epsilon=epsilon, k=20, param_alpha=0.5)

print("Running TwinGreedy2...")
run_algorithm("TwinGreedy2", twingreedy2, gnd, f_diff, ind_add_oracle, ind_oracle, knapsack_constraints, budget1,
              budget2, rating_array, date_array, max_rating, year1)

print("Running Algorithm3ITW...")
run_algorithm("Algorithm3ITW", algorithm3_itw, gnd, f_diff, ind_add_oracle, ind_oracle, knapsack_constraints,
              budget1, budget2, rating_array, date_array, max_rating, year1, mu=1.0)


print("All algorithms completed. Results are saved in algorithm_results.csv")


