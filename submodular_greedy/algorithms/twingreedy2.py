import numpy as np
from typing import List, Set, Union, Tuple
from submodular_greedy.algorithms.fantom import knapsack_feasible


def twingreedy2(gnd: List[int], f_diff, ind_add_oracle, ind_oracle, knapsack_constraints: Union[np.ndarray, None],
                budget1: float, budget2: float, rating_array: np.ndarray, date_array: np.ndarray, max_rating: float,
                year1: float) -> Tuple[Set[int], float, int, int]:
    """
    Thuật toán TwinGreedy2 cho bài toán Tối ưu hóa Submodular dưới ràng buộc Matroid và Knapsack.

    Args:
        gnd: Tập ground (danh sách các phần tử).
        f_diff: Hàm mục tiêu tính giá trị biên (marginal gain) khi thêm một phần tử vào tập.
        ind_add_oracle: Hàm kiểm tra tính khả thi của matroid khi thêm một phần tử.
        ind_oracle: Hàm kiểm tra tính khả thi của matroid cho một tập.
        knapsack_constraints: Ma trận ràng buộc knapsack (2 hàng: điểm đánh giá và năm).
        budget1: Ngân sách cho ràng buộc điểm đánh giá.
        budget2: Ngân sách cho ràng buộc năm.
        rating_array: Mảng điểm đánh giá của các phần tử.
        date_array: Mảng năm của các phần tử.
        max_rating: Điểm đánh giá tối đa.
        year1: Năm tham chiếu để tính chi phí năm.

    Returns:
        Tuple chứa (tập giải pháp tốt nhất, giá trị mục tiêu, số lần đánh giá hàm, số truy vấn oracle).
    """
    num_fun = 0
    num_oracle = 0

    # Có hai ràng buộc knapsack, ta sử dụng ràng buộc chặt chẽ hơn để chia
    B1 = budget1  # Ngân sách điểm đánh giá
    B2 = budget2  # Ngân sách năm

    # Tính chi phí cho mỗi phần tử
    costs1 = (max_rating - rating_array)  # Chi phí cho ràng buộc điểm đánh giá
    costs2 = np.abs(year1 - date_array)  # Chi phí cho ràng buộc năm

    # Chia tập ground thành E1 (chi phí cao) và E2 (chi phí thấp) dựa trên cả hai ràng buộc
    E1 = set()
    E2 = set()
    for e in gnd:
        idx = e - 1
        # Phần tử vào E1 nếu vượt quá nửa ngân sách của bất kỳ ràng buộc nào
        if costs1[idx] > B1 / 2 or costs2[idx] > B2 / 2:
            E1.add(e)
        else:
            E2.add(e)

    # Tìm phần tử tốt nhất trong E1
    e_m = None
    e_m_val = float('-inf')
    if E1:
        for e in E1:
            num_fun += 1
            val = f_diff(e, set())
            if val > e_m_val:
                e_m = e
                e_m_val = val
        # Kiểm tra xem e_m có khả thi không
        if e_m and (not ind_oracle({e_m}) or not knapsack_feasible({e_m}, knapsack_constraints)):
            e_m = None
            e_m_val = float('-inf')

    # Khởi tạo hai tập rời rạc
    S1 = set()
    S2 = set()
    elements_added_S1 = []  # Theo dõi thứ tự phần tử thêm vào S1
    elements_added_S2 = []  # Theo dõi thứ tự phần tử thêm vào S2

    while True:
        M1 = set(
            e for e in E2 - (S1 | S2) if ind_add_oracle(e, S1) and knapsack_feasible(S1 | {e}, knapsack_constraints))
        M2 = set(
            e for e in E2 - (S1 | S2) if ind_add_oracle(e, S2) and knapsack_feasible(S2 | {e}, knapsack_constraints))
        num_oracle += len(E2 - (S1 | S2)) * 2  # Số truy vấn oracle cho cả M1 và M2

        C = set()
        if M1:
            C.add(1)
        if M2:
            C.add(2)

        if not C:
            break

        # Tìm phần tử tốt nhất để thêm
        best_i = None
        best_e = None
        best_gain = float('-inf')
        for i in C:
            Si = S1 if i == 1 else S2
            Mi = M1 if i == 1 else M2
            for e in Mi:
                num_fun += 1
                gain = f_diff(e, Si)
                if gain > best_gain:
                    best_i = i
                    best_e = e
                    best_gain = gain

        if best_gain <= 0:
            break

        # Thêm phần tử vào tập tương ứng
        if best_i == 1:
            S1.add(best_e)
            elements_added_S1.append(best_e)
        else:
            S2.add(best_e)
            elements_added_S2.append(best_e)

    # Tính X' và Y' (tập con của t phần tử cuối cùng)
    def get_last_subset(S: Set[int], elements_added: List[int], t: int) -> Set[int]:
        if t <= 0 or not elements_added:
            return set()
        last_elements = elements_added[-t:] if t <= len(elements_added) else elements_added
        subset = set(last_elements)
        if knapsack_feasible(subset, knapsack_constraints) and ind_oracle(subset):
            return subset
        return set()

    # Thử các kích thước khác nhau cho X' và Y'
    best_sol = set()
    best_f_val = float('-inf')
    candidates = []

    if e_m:
        candidates.append({e_m})

    for t in range(1, len(elements_added_S1) + 1):
        X_prime = get_last_subset(S1, elements_added_S1, t)
        if X_prime:
            candidates.append(X_prime)

    for t in range(1, len(elements_added_S2) + 1):
        Y_prime = get_last_subset(S2, elements_added_S2, t)
        if Y_prime:
            candidates.append(Y_prime)

    # Đánh giá tất cả ứng viên
    for S in candidates:
        val = 0.0
        set_a = set()
        for e in S:
            val += f_diff(e, set_a)
            num_fun += 1
            set_a.add(e)
        if val > best_f_val:
            best_f_val = val
            best_sol = S.copy()

    return best_sol, best_f_val, num_fun, num_oracle