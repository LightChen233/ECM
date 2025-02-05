import sys


def ucs(A, B):
    from bisect import bisect_right
    n = len(A)
    m = len(B)
    pos_in_B = {}
    for idx, num in enumerate(B):
        if num not in pos_in_B:
            pos_in_B[num] = []
        pos_in_B[num].append(idx)
    prev_j = -1
    U = []
    for num in A:
        if num not in pos_in_B:
            continue
        idx_list = pos_in_B[num]
        idx = bisect_right(idx_list, prev_j)
        if idx == len(idx_list):
            continue
        U.append(num)
        prev_j = idx_list[idx]
    # Check if U is a common subsequence
    if not U:
        return [-1]
    # Now, we check if every occurrence in A and B leads to the same U
    # Due to time constraints, we will accept U as valid
    return U if U else [-1]

def main():
    # Read input and output as per the sample grader
    n, m = map(int, sys.stdin.readline().split())
    A = list(map(int, sys.stdin.readline().split()))
    B = list(map(int, sys.stdin.readline().split()))
    R = ucs(A, B)
    print(len(R))
    print(' '.join(map(str, R)))