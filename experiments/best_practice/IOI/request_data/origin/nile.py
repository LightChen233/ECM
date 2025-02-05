from io import StringIO
import sys
from typing import List
import sys

class DSU:
    def __init__(self, A: List[int], B: List[int]):
        n = len(A)
        self.parent = [-1] * n
        self.size = [1] * n
        self.even = [a - b for a, b in zip(A, B)]
        self.odd = [sys.maxsize] * n
        self.cost = sum(self.even)

    def get_par(self, x: int) -> int:
        while self.parent[x] != -1:
            if self.parent[self.parent[x]] != -1:
                self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def merge(self, v: int):
        u = self.get_par(v - 1)
        for x in (u, v):
            if self.size[x] % 2 == 1:
                self.cost -= self.even[x]

        if self.size[u] % 2 == 1:
            self.even[v], self.odd[v] = self.odd[v], self.even[v]
        
        self.odd[u] = min(self.odd[u], self.odd[v])
        self.even[u] = min(self.even[u], self.even[v])

        self.parent[v] = u
        self.size[u] += self.size[v]
        if self.size[u] % 2 == 1:
            self.cost += self.even[u]

    def consider(self, v: int, w: int):
        u = self.get_par(v)
        if (v - u) % 2 == 1:
            if self.size[u] % 2 == 1:
                self.cost -= self.even[u]
            self.even[u] = min(self.even[u], w)
            if self.size[u] % 2 == 1:
                self.cost += self.even[u]
        else:
            self.odd[u] = min(self.odd[u], w)

def calculate_costs(W: List[int], A: List[int], B: List[int], E: List[int]) -> List[int]:
    n = len(W)
    q = len(E)
    
    if n == 1:
        return [A[0]] * q

    # Create a helper list with tuples of (W, A, B) and sort it
    helper = sorted(zip(W, A, B))
    W, A, B = zip(*helper)

    # Prepare difference arrays and sort indices based on their conditions
    diff1 = sorted(range(1, n), key=lambda i: W[i] - W[i - 1])
    diff2 = sorted(range(1, n - 1), key=lambda i: W[i + 1] - W[i - 1])
    indexes = sorted(range(q), key=lambda i: E[i])

    dsu = DSU(A, B)
    p1 = 0
    p2 = 0
    costs = [0] * q
    total = sum(B)
    
    for index in indexes:
        while p1 < n - 1 and W[diff1[p1]] - W[diff1[p1] - 1] <= E[index]:
            idx = diff1[p1]
            p1 += 1
            dsu.merge(idx)

            for p in (idx, idx - 1):
                if 0 < p < n - 1 and W[p + 1] - W[p - 1] <= E[index] and dsu.get_par(p + 1) == dsu.get_par(p - 1):
                    dsu.consider(p, A[p] - B[p])
        
        while p2 < n - 2 and W[diff2[p2] + 1] - W[diff2[p2] - 1] <= E[index]:
            idx = diff2[p2]
            p2 += 1
            if dsu.get_par(idx + 1) == dsu.get_par(idx - 1):
                dsu.consider(idx, A[idx] - B[idx])
        
        costs[index] = total + dsu.cost

    return costs

# Sample usage with the provided test case
# N = 5
# W = [15, 12, 2, 10, 21]
# A = [5, 4, 5, 6, 3]
# B = [1, 2, 2, 3, 2]
# E = [5, 9, 1]
def main():
    N = int(input())
    W, A, B = [], [], []
    for _ in range(N):
        w, a, b = map(int, input().split())
        W.append(w)
        A.append(a)
        B.append(b)
    Q = int(input())
    E = [int(input()) for _ in range(Q)]
    results = calculate_costs(W, A, B, E)
    for res in results:
        print(res)

