import random
import numpy as np
import pandas as pd
from typing import Set, List, Union, Tuple, Dict
from itertools import combinations
from scipy.spatial.distance import euclidean
from submodular_greedy import greedy, repeated_greedy, simultaneous_greedys, fantom, main_part_sprout

random.seed(12)

# Đọc dữ liệu từ file CSV (giả lập)
data_file = "../../dataset/movie_info.csv"
movie_info_df = pd.read_csv(data_file)

# Chuyển đổi cột vec từ chuỗi sang danh sách
movie_info_df['vec'] = movie_info_df['vec'].apply(lambda x: [float(v) for v in x.strip('[]').split(',')])
# Chuyển đổi cột genre_set từ chuỗi sang tập hợp
movie_info_df['genre_set'] = movie_info_df['genre_set'].apply(lambda x: set(x.strip('{}').split(', ')))

n = len(movie_info_df)

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

similarity_array = compute_similarity_array(movie_info_df['vec'].tolist())

def dispersion_diff(elm: int, sol: Set[int], similarity_array: np.ndarray, n: int) -> float:
    if elm in sol:
        return 0.0
    else:
        coverage_term = sum(get_similarity(similarity_array, i + 1, elm + 1, n) for i in range(n))
        diversity_term = (2 * sum(get_similarity(similarity_array, i + 1, elm + 1, n) for i in sol) +
                          get_similarity(similarity_array, elm + 1, elm + 1, n))
        return (coverage_term - diversity_term) / n

def f_diff(elm: int, sol: Set[int]) -> float:
    return dispersion_diff(elm, sol, similarity_array, n)

# Bước 2: Xây dựng các ràng buộc
# Ràng buộc matroid
movie_genre_df = movie_info_df['genre_set'].tolist()
card_limit = 10

genre_list = set()
for genres in movie_genre_df:
    genre_list.update(genres)
genre_list = list(genre_list)

genre_card_limit = 2
limit_values = [genre_card_limit] * 19
genre_limit = {genre_list[i]: limit_values[i] for i in range(len(limit_values))}

def all_matroid_feasible(sol: Set[int], cardinality_limit: int, genre_limit: Dict[str, int],
                         genre_list: List[str], movie_genre_df: List[Set[str]]) -> bool:
    if len(sol) > cardinality_limit:
        return False
    genre_count = {genre: 0 for genre in genre_list}
    for elm in sol:
        for genre in movie_genre_df[elm]:
            if genre in genre_list:
                genre_count[genre] += 1
                if genre_count[genre] > genre_limit[genre]:
                    return False
    return True

def ind_add_oracle(elm: int, sol: Set[int]) -> bool:
    return all_matroid_feasible(sol.union({elm}), card_limit, genre_limit, genre_list, movie_genre_df)

def ind_oracle(sol: Set[int]) -> bool:
    return all_matroid_feasible(sol, card_limit, genre_limit, genre_list, movie_genre_df)

# Ràng buộc knapsack
budget_param = 1.0
knapsack_constraints = np.zeros((2, n))

rating_array = movie_info_df['rating'].to_numpy()
max_rating = 10
budget1 = 20 * budget_param
knapsack_constraints[0, :] = (max_rating - rating_array) / budget1

date_array = movie_info_df['year'].to_numpy()
year1 = 1995
budget2 = 30 * budget_param
knapsack_constraints[1, :] = np.abs(year1 - date_array) / budget2

# Chạy các thuật toán
gnd = list(range(n))

# 1: Greedy
sol, f_val, _, _, _ = greedy(gnd, f_diff, ind_add_oracle, knapsack_constraints=knapsack_constraints)
print("Greedy:", sol, f_val)

# 2: Repeated Greedy
num_sol = 2
epsilon = 0.25
sol, f_val, _, _ = repeated_greedy(gnd, f_diff, ind_add_oracle, knapsack_constraints=knapsack_constraints,
                                    num_sol=num_sol, epsilon=epsilon, k=20)
print("Repeated Greedy:", sol, f_val)

# 3: Simultaneous Greedy
sol, f_val, _, _ = simultaneous_greedys(gnd, f_diff, ind_add_oracle, knapsack_constraints=knapsack_constraints,
                                        extendible=True, num_sol=num_sol, epsilon=epsilon, k=20)
print("Simultaneous Greedy:", sol, f_val)

# 4: FANTOM
sol, f_val, _, _ = fantom(gnd, f_diff, ind_add_oracle, knapsack_constraints=knapsack_constraints,
                          epsilon=epsilon, k=20)
print("FANTOM:", sol, f_val)

# 5: SPROUT++
tc = int(np.floor(n * 0.0005))
param_c = 1
param_alpha = 0.5
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
        if cnt >= tc:
            break
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

        print(f"--------------------------------{cnt} iteration--------------------------------")
        cnt += 1

        def z_diff(elm: int, sol: Set[int]) -> float:
            return f_diff(elm, sol.union(gnd_combinatorial))
        num_fun += 1

        gnd_new = [x for x in gnd if x not in gnd_combinatorial]
        card_limit_new = card_limit - param_c
        genre_limit_new = genre_limit.copy()

        for elm in gnd_combinatorial:
            for genre in movie_genre_df[elm]:
                if genre in genre_list:
                    genre_limit_new[genre] = max(genre_limit_new[genre] - 1, 0)

        def ind_add_oracle_new(elm: int, sol: Set[int]) -> bool:
            return all_matroid_feasible(sol.union({elm}), card_limit_new, genre_limit_new, genre_list, movie_genre_df)

        knapsack_constraints_new = knapsack_constraints.copy()
        budget1_new = budget1 - np.sum((max_rating - rating_array)[list(gnd_combinatorial)])
        knapsack_constraints_new[0, :] = (max_rating - rating_array) / budget1_new

        budget2_new = budget2 - np.sum(np.abs(year1 - date_array)[list(gnd_combinatorial)])
        knapsack_constraints_new[1, :] = np.abs(year1 - date_array) / budget2_new

        sol, f_val, oracle_calls, _ = main_part_sprout(
            param_c, fA_value, gnd_new, z_diff, ind_add_oracle_new, knapsack_constraints=knapsack_constraints_new,
            extendible=True, num_sol=num_sol, epsilon=epsilon, k=20, mu=mu)
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

    return best_sol, best_f_val, num_fun

sol, f_val, _ = sproutpp(param_c, gnd, f_diff, ind_oracle, ind_add_oracle, knapsack_constraints=knapsack_constraints,
                         extendible=True, num_sol=num_sol, epsilon=epsilon, k=20, param_alpha=0.5)
print("SPROUT++:", sol, f_val)