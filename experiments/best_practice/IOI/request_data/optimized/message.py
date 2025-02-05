from io import StringIO
import os
import sys

# Aisha's send_message function
def send_message(M, C):
    plen = 31
    maxe = 16
    A = [[False] * plen for _ in range(max(maxe, (len(M) + 15) // 16 + 2))]
    nxt = [1] * plen

    for b in range(plen):
        if not C[b]:
            while C[(b + nxt[b]) % plen]:
                nxt[b] += 1

    for b in range(plen):
        if not C[b]:
            A[nxt[b] - 1][b] = True

    pos = 0
    for i in range(len(A)):
        for b in range(plen):
            if not C[b] and nxt[b] <= i:
                A[i][b] = M[pos] if pos < len(M) else (pos == len(M))
                pos += 1

    for i in range(len(A)):
        send_packet(A[i])

def receive_message(A):
    plen = 31
    maxe = 16
    nxt = [0] * plen

    for b in range(plen):
        for i in range(maxe):
            if A[i][b]:
                nxt[b] = i + 1
                break

    C = [True] * plen
    for b in range(plen):
        cnt = 0
        mark = [False] * plen
        v = (b + nxt[b]) % plen
        while not mark[v]:
            mark[v] = True
            v = (v + nxt[v]) % plen
            cnt += 1

        if cnt == 16:
            for v in range(plen):
                C[v] = not mark[v]

    M = []
    for i in range(len(A)):
        for b in range(plen):
            if not C[b] and nxt[b] <= i:
                M.append(A[i][b])

    while not M[-1]:
        M.pop()

    M.pop()
    return M


PACKET_SIZE = 31
CALLS_CNT_LIMIT = 100

calls_cnt = 0
C = [False] * PACKET_SIZE
R = []

def quit_program(message):
    print(message)
    sys.exit(0)

def run_scenario():
    global calls_cnt, R
    R.clear()
    calls_cnt = 0

    S = int(input().strip())
    M = [bool(int(x.strip())) for x in input().split(" ")][:S]
    input_list = input().split(" ")
    idx = 0
    for i in range(PACKET_SIZE):
        bit = int(input_list[idx])
        assert bit in (0, 1)
        C[i] = bool(bit)
        idx += 1
    # print(M, C)
    send_message(M, C)
    # print(R)
    D = receive_message(R)

    K = len(R)
    L = len(D)
    print(f"{K} {L}")
    print(" ".join(str(int(bit)) for bit in D))

def taint(A):
    B = A[:]
    bit = False
    for i in range(PACKET_SIZE):
        if C[i]:
            B[i] = bit
            bit = not bit
    return B

def send_packet(A):
    global calls_cnt,R
    calls_cnt += 1
    if calls_cnt > CALLS_CNT_LIMIT:
        quit_program("Too many calls")
    if len(A) != PACKET_SIZE:
        quit_program("Invalid argument")

    B = taint(A)
    R.append(B)
    return B


    

def main():
    T = int(input().strip())
    for _ in range(T):
        run_scenario()

