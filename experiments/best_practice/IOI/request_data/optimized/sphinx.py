from collections import defaultdict
from typing import List
import sys
from collections import deque

N = 0
p = [-1] * 250
edge = defaultdict(list)
ord = []
reached = [False] * 250
unchecked = set()
keep = [False] * 250
ccEdge = [set() for _ in range(250)]
cc = set()
parity = [0] * 250

def where(x: int) -> int:
    if p[x] < 0:
        return x
    p[x] = where(p[x])
    return p[x]

def color(x: int) -> int:
    return ord[x] + N if ord[x] >= 0 else where(x)

def dfs(x: int):
    reached[x] = True
    for i in edge[x]:
        if not reached[i] and color(x) == color(i):
            dfs(i)

def expected() -> int:
    for i in range(N):
        reached[i] = False
    
    sum_ = 0
    for i in range(N):
        if not reached[i]:
            sum_ += 1
            dfs(i)
    
    return sum_

def pDfs(x: int):
    reached[x] = True
    for i in ccEdge[x]:
        if not reached[i]:
            parity[i] = 1 - parity[x]
            pDfs(i)

def find_colours(NN: int, X: List[int], Y: List[int]) -> List[int]:
    global N, p, edge, ord, ccEdge, parity, reached, keep, cc
    N = NN
    ord = [-1] * N
    p = [-1] * N
    edge = defaultdict(list)
    ccEdge = [set() for _ in range(250)]
    parity = [0] * 250
    reached = [False] * 250
    keep = [False] * 250
    cc = set()
    
    M = len(X)
    for i in range(M):
        edge[Y[i]].append(X[i])
        edge[X[i]].append(Y[i])
    
    for i in range(1, N):
        while True:
            ord = [-1] * (i + 1) + [N] * (N - (i + 1))
            if perform_experiment(ord) == expected():
                break
            
            unchecked.clear()
            for j in range(i):
                if where(j) != i:
                    unchecked.add(where(j))
            
            vec = list(unchecked)
            a, b = 0, len(vec) - 1
            while a != b:
                half = (a + b) // 2
                keep = [False] * N
                for j in range(a, half + 1):
                    keep[vec[j]] = True
                for j in range(i):
                    ord[j] = -1 if keep[where(j)] else N
                ord[i] = -1
                for j in range(i + 1, N):
                    ord[j] = N

                if perform_experiment(ord) == expected():
                    a = half + 1
                else:
                    b = half
            
            p[where(vec[a])] = i
    
    for i in range(N):
        cc.add(where(i))
    
    F = [-1] * N
    if len(cc) == 1:
        for i in range(N):
            ord = [-1] * N
            ord[0] = i
            if perform_experiment(ord) == 1:
                F = [i] * N
                break
    else:
        for i in range(N):
            for j in edge[i]:
                if where(i) != where(j):
                    ccEdge[where(i)].add(where(j))
                    ccEdge[where(j)].add(where(i))
        
        for i in range(N):
            reached[i] = False
        pDfs(next(iter(cc)))
        
        for par in range(2):
            for i in range(N):
                while True:
                    for j in range(N):
                        if parity[where(j)] == par or F[where(j)] != -1:
                            ord[j] = i
                        else:
                            ord[j] = -1
                    
                    if perform_experiment(ord) == expected():
                        break
                    
                    unchecked.clear()
                    for j in range(N):
                        if parity[where(j)] == 1 - par and F[where(j)] == -1:
                            unchecked.add(where(j))
                    
                    vec = list(unchecked)
                    a, b = 0, len(vec) - 1
                    while a != b:
                        half = (a + b) // 2
                        keep = [False] * N
                        for j in range(a, half + 1):
                            if F[vec[j]] == -1:
                                keep[vec[j]] = True
                        for j in range(N):
                            ord[j] = -1 if keep[where(j)] else i
                        
                        if perform_experiment(ord) == expected():
                            a = half + 1
                        else:
                            b = half
                    
                    F[where(vec[a])] = i
    
    for i in range(N):
        F[i] = F[where(i)]
    
    return F




# Constants
CALLS_CNT_LIMIT = 2750

# Variables
calls_cnt = 0
N = 0
M = 0
C = []
X = []
Y = []
adj = []

def quit_program(message):
    print(message)
    sys.exit(0)

def count_components(S):
    components_cnt = 0
    vis = [False] * N
    for i in range(N):
        if vis[i]:
            continue
        components_cnt += 1

        q = deque()
        vis[i] = True
        q.append(i)
        while q:
            cur = q.popleft()
            for nxt in adj[cur]:
                if not vis[nxt] and S[nxt] == S[cur]:
                    vis[nxt] = True
                    q.append(nxt)
    return components_cnt

def perform_experiment(E):
    global calls_cnt
    calls_cnt += 1
    if calls_cnt > CALLS_CNT_LIMIT:
        quit_program("Too many calls")
    if len(E) != N:
        quit_program("Invalid argument")
    for e in E:
        if not (-1 <= e <= N):
            quit_program("Invalid argument")

    S = [C[i] if E[i] == -1 else E[i] for i in range(N)]
    return count_components(S)

def main():
    global N, M, C, X, Y, adj, calls_cnt
    
    # Reading input
    N, M = map(int, input().split())
    C = list(map(int, input().split()))
    X = [0] * M
    Y = [0] * M
    for j in range(M):
        X[j], Y[j] = map(int, input().split())
    
    # Construct adjacency list
    adj = [[] for _ in range(N)]
    for j in range(M):
        adj[X[j]].append(Y[j])
        adj[Y[j]].append(X[j])
    
    # Reset the call count and perform the experiment
    calls_cnt = 0
    G = find_colours(N, X, Y)

    # Output the result
    L = len(G)
    print(f"{L} {calls_cnt}")
    print(" ".join(map(str, G)))
    return
