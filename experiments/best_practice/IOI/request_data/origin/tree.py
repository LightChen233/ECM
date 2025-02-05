from typing import List

class Tree:
    def __init__(self):
        self.fa = []
        self.n = 0
        self.sumleaf = 0
        self.ans = []

    def find(self, u: int) -> int:
        if self.fa[u] == u:
            return u
        self.fa[u] = self.find(self.fa[u])
        return self.fa[u]

    def query(self, l: int, r: int) -> int:
        k = r // l
        if k > self.n:
            res = 0
        else:
            res = l * self.ans[k][0] + r * self.ans[k][1]
        return res + l * self.sumleaf

    def init(self, anc: List[int], wei: List[int]):
        self.n = len(anc)
        g = [[] for _ in range(self.n)]
        for i in range(1, self.n):
            g[anc[i]].append(i)
        
        self.sumleaf = 0
        for i in range(self.n):
            if len(g[i]) == 0:
                self.sumleaf += wei[i]
                wei[i] = 0

        self.ans = [[0, 0] for _ in range(self.n + 2)]

        def upd(k: int, w: int):
            self.ans[k - 1][0] += k * w
            self.ans[k - 1][1] -= w

        eff = sorted(((wei[i], i) for i in range(self.n)), reverse=True)

        self.fa = list(range(self.n))
        num = [1] * self.n
        rig = [eff[0][0]] * self.n
        val = [0] * self.n

        def connect(u: int, v: int, cur: int):
            u = self.find(u)
            v = self.find(v)
            if u == v:
                return
            upd(num[u], rig[u] - cur)
            upd(num[v], rig[v] - cur)
            num[v] += num[u]
            rig[v] = cur
            self.fa[u] = v

        for t in eff:
            if t[0] == 0:
                break

            u = t[1]
            val[u] = 1

            if anc[u] >= 0 and val[anc[u]]:
                connect(u, anc[u], t[0])
            for v in g[u]:
                connect(v, u, t[0])

            v = self.find(u)
            if rig[v] != t[0]:
                upd(num[v], rig[v] - t[0])
            num[v] -= 1
            rig[v] = t[0]

        for i in range(self.n):
            if self.fa[i] == i:
                upd(num[i], rig[i])

        for i in range(self.n, 0, -1):
            self.ans[i][0] += self.ans[i + 1][0]
            self.ans[i][1] += self.ans[i + 1][1]

    def tree(self, anc: List[int], wei: List[int], L: List[int], R: List[int]) -> List[int]:
        self.init(anc, wei)
        res = [self.query(l, r) for l, r in zip(L, R)]
        return res
exit_allowed = True

def main():
    N = int(input())
    P = [-1] * N
    P[1:] = map(int, input().split())

    W = list(map(int, input().split()))

    Q = int(input())
    L = [0] * Q
    R = [0] * Q
    for j in range(Q):
        L[j], R[j] = map(int, input().split())

    # BEGIN SECRET
    exit_allowed = False
    # END SECRET
    tree = Tree()
    tree.init(P, W)
    A = [tree.query(L[j], R[j]) for j in range(Q)]

    # END SECRET

    for result in A:
        print(result)