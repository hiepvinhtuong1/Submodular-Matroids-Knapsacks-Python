import sys
import os

# Thêm thư mục gốc vào Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, project_root)

import time
import pandas as pd
import numpy as np
from typing import List, Set, Union, Tuple
from scipy.spatial.distance import euclidean
from itertools import combinations
import random
import ast
from submodular_greedy import greedy, repeated_greedy, simultaneous_greedys, fantom, main_part_sprout, twingreedy2,algorithm3_itw
from submodular_greedy.algorithms.fantom import knapsack_feasible

# Đọc dữ liệu từ file CSV
data_file = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "movie_info5000.csv")
movie_info_df = pd.read_csv(data_file)

n = len(movie_info_df)
print(movie_info_df.columns)


# Bước 1: Xây dựng hàm mục tiêu
def ij_to_ind(i: int, j: int, n: int) -> int:
    if i > j:
        i, j = j, i
    return (i - 1) * n - (i * (i - 1)) // 2 + j


def get_similarity(similarity_array: np.ndarray, i: int, j: int, n: int) -> float:
    ind = ij_to_ind(i, j, n)
    return similarity_array[ind - 1]


def compute_similarity_array(vec_list: List[List[float]], sigma: float = 1.0) -> np.ndarray:
    n = len(vec_list)
    n_pairs = (n * (n + 1)) // 2
    similarity_array = np.zeros(n_pairs)

    ind = 0
    for i in range(n):
        vi = vec_list[i]
        for j in range(i, n):
            vj = vec_list[j]
            similarity_array[ind] = np.exp(-4 * euclidean(vi, vj))
            ind += 1
    return similarity_array


# Phân tích cột vec từ chuỗi sang danh sách
movie_info_df['vec'] = movie_info_df['vec'].apply(ast.literal_eval)
similarity_array = compute_similarity_array(movie_info_df['vec'].tolist())


def dispersion_diff(elm: int, sol: Set[int], similarity_array: np.ndarray, n: int) -> float:
    if elm in sol:
        return 0.0
    coverage_term = sum(get_similarity(similarity_array, i, elm, n) for i in range(1, n + 1))
    diversity_term = (
            2 * sum(get_similarity(similarity_array, i, elm, n) for i in sol) + get_similarity(similarity_array, elm,
                                                                                               elm, n))
    return (coverage_term - diversity_term) / n


def f_diff(elm: int, sol: Set[int]) -> float:
    return dispersion_diff(elm, sol, similarity_array, n)


# Bước 2: Xây dựng ràng buộc
# Phân tích genre_set từ chuỗi sang tập
movie_info_df['genre_set'] = movie_info_df['genre_set'].apply(
    lambda x: set(ast.literal_eval(x.replace('Set([', '[').replace('])', ']'))))
movie_genre_df = movie_info_df['genre_set'].tolist()

card_limit = 10

genre_list = set()
for genres in movie_genre_df:
    genre_list.update(genres)
genre_list = sorted(list(genre_list))

genre_card_limit = 2
limit_values = [genre_card_limit] * 19

genre_limit = {genre: limit for genre, limit in zip(genre_list, limit_values)}


def all_matroid_feasible(sol: Set[int], cardinality_limit: int, genre_limit: dict, genre_list: List[str],
                         movie_genre_df: List[Set[str]]) -> bool:
    if len(sol) > cardinality_limit:
        return False

    genre_count = {genre: 0 for genre in genre_list}
    for elm in sol:
        for genre in movie_genre_df[elm - 1]:
            if genre in genre_list:
                genre_count[genre] += 1
                if genre_count[genre] > genre_limit[genre]:
                    return False
    return True


ind_add_oracle = lambda elm, sol: all_matroid_feasible(sol | {elm}, card_limit, genre_limit, genre_list, movie_genre_df)
ind_oracle = lambda sol: all_matroid_feasible(sol, card_limit, genre_limit, genre_list, movie_genre_df)

budget_param = 1

# Xây dựng ràng buộc knapsack
knapsack_constraints = np.zeros((2, n))
rating_array = movie_info_df['rating'].to_numpy()
max_rating = 10
budget1 = 20 * budget_param
knapsack_constraints[0, :] = (max_rating - rating_array) / budget1

date_array = movie_info_df['year'].to_numpy()
year1 = 1995
budget2 = 30 * budget_param
knapsack_constraints[1, :] = np.abs(year1 - date_array) / budget2

# Hàm SPROUT++ (giữ nguyên từ mã gốc)
tc = 2000
param_c = 1
param_alpha = 0.5
mu = 1.0


def sproutpp(param_c: int, gnd: List[int], f_diff, ind_oracle, ind_add_oracle, num_sol: int = 0, k: int = 0,
             knapsack_constraints: Union[np.ndarray, None] = None, extendible: bool = False, monotone: bool = False,
             epsilon: float = 0.0, opt_size_ub: int = None, verbose_lvl: int = 1, param_alpha: float = 0.5) -> Tuple[
    Set[int], float, int]:
    num_fun = 0
    best_sol = None
    best_f_val = 0

    gnd_combinatorials = list(combinations(gnd, param_c))
    random.shuffle(gnd_combinatorials)

    fA_max = 0.0
    for elm in gnd:
        fA_max = max(fA_max, f_diff(elm, set()))
    num_fun += len(gnd)

    cnt = 0
    for gnd_combinatorial in gnd_combinatorials:
        gnd_combinatorial = set(gnd_combinatorial)
        if not ind_oracle(gnd_combinatorial) or not knapsack_feasible(gnd_combinatorial, knapsack_constraints):
            continue

        fA_value = 0.0
        set_a = set()
        for elm in gnd_combinatorial:
            fA_value += f_diff(elm, set_a)
            num_fun += 1
            set_a.add(elm)

        if fA_value < param_alpha * fA_max:
            continue
        print("--------------------------------", cnt, " iteration--------------------------------")
        if cnt >= tc:
            break
        cnt += 1

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
        adjusted_indices = [i - 1 for i in list(gnd_combinatorial)]
        budget1_new = budget1 - np.sum((max_rating - rating_array[adjusted_indices]))
        budget2_new = budget2 - np.sum(np.abs(year1 - date_array[adjusted_indices]))

        if budget1_new <= 0 or budget2_new <= 0:
            continue

        knapsack_constraints_new[0, :] = (max_rating - rating_array) / budget1_new
        knapsack_constraints_new[1, :] = np.abs(year1 - date_array) / budget2_new

        sol, f_val, oracle_calls, _ = main_part_sprout(param_c, fA_value, gnd_new, z_diff, ind_add_oracle_new,
                                                       knapsack_constraints=knapsack_constraints_new, extendible=True,
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


# Helper function to run an algorithm and collect metrics
def run_algorithm(name, func, *args, **kwargs):
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()

    # Extract results based on function signature
    if name == "SPROUT++":
        sol, f_val, num_fun = result
        num_oracle = None
    else:
        sol, f_val, num_fun, num_oracle = result[:4]
        if len(result) > 4:
            knap_reject = result[4]

    running_time = end_time - start_time
    result_dict = {
        "Algorithm": name,
        "budget": budget_param,
        "f(S)": f_val,
        "Number of query": num_fun,
        "running time": running_time
    }

    # Lưu kết quả vào file CSV
    result_df = pd.DataFrame([result_dict])
    output_file = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data",
                               "algorithm_results.csv")

    # Kiểm tra xem file đã tồn tại chưa
    if not os.path.exists(output_file):
        result_df.to_csv(output_file, index=False, mode='w')
    else:
        result_df.to_csv(output_file, index=False, mode='a', header=False)

    print(f"Saved result for {name} to {output_file}")

    return result_dict


# Chạy các thuật toán và lưu kết quả
gnd = list(range(1, n + 1))
num_sol = 2
epsilon = 0.25

# List to store results (vẫn giữ để in ra màn hình cuối cùng)
results = []

# Run each algorithm and save to CSV immediately
# 1: Greedy
results.append(run_algorithm("Greedy", greedy, gnd, f_diff, ind_add_oracle, knapsack_constraints=knapsack_constraints))

# 2: Repeated Greedy
results.append(run_algorithm("Repeated Greedy", repeated_greedy, gnd, f_diff, ind_add_oracle,
                             knapsack_constraints=knapsack_constraints, num_sol=num_sol, epsilon=epsilon, k=20))

# 3: Simultaneous Greedy
results.append(run_algorithm("Simultaneous Greedy", simultaneous_greedys, gnd, f_diff, ind_add_oracle,
                             knapsack_constraints=knapsack_constraints, extendible=True, num_sol=num_sol,
                             epsilon=epsilon, k=20))

# 4: FANTOM
results.append(run_algorithm("FANTOM", fantom, gnd, f_diff, ind_add_oracle, knapsack_constraints=knapsack_constraints,
                             epsilon=epsilon, k=20))

# 5: SPROUT++
results.append(run_algorithm("SPROUT++", sproutpp, param_c, gnd, f_diff, ind_oracle, ind_add_oracle,
                             knapsack_constraints=knapsack_constraints, extendible=True, num_sol=num_sol,
                             epsilon=epsilon, k=20, param_alpha=0.5))

# 6: TwinGreedy2
results.append(
    run_algorithm("TwinGreedy2", twingreedy2, gnd, f_diff, ind_add_oracle, ind_oracle, knapsack_constraints, budget1,
                  budget2, rating_array, date_array, max_rating, year1))

# 7: Algorithm3ITW
results.append(
    run_algorithm("Algorithm3ITW", algorithm3_itw, gnd, f_diff, ind_add_oracle, ind_oracle, knapsack_constraints,
                  budget1, budget2, rating_array, date_array, max_rating, year1, mu=1.0))

# Convert results to DataFrame for final display
results_df = pd.DataFrame(results)

print(f"Final results:")
print(results_df)