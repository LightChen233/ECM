"""4
1 0 1 0
1 1 0 1
2
0 3 0 3
2 3 0 2
"""

def mosaic(X, Y, T, B, L, R):
    N = len(X)
    Q = len(T)
    # Initialize the grid colors with zeros
    grid_color = [ [0]*N for _ in range(N) ]
    # Precompute the grid colors using the formula
    for i in range(N):
        for j in range(N):
            p = (i + j) % 2
            if p == 0:
                grid_color[i][j] = X[j] ^ Y[i]
            else:
                grid_color[i][j] = 1 ^ X[j] ^ Y[i]
    # Build prefix sums for the grid to answer queries efficiently
    prefix_sum = [ [0]*(N+1) for _ in range(N+1) ]
    for i in range(N):
        for j in range(N):
            prefix_sum[i+1][j+1] = grid_color[i][j] + prefix_sum[i][j+1] + prefix_sum[i+1][j] - prefix_sum[i][j]
    # Answer the queries
    result = []
    for k in range(Q):
        top = T[k]
        bottom = B[k]
        left = L[k]
        right = R[k]
        total = prefix_sum[bottom+1][right+1] - prefix_sum[bottom+1][left] - prefix_sum[top][right+1] + prefix_sum[top][left]
        result.append(total)
    return result

# I/O Functionality
def main():
    import sys

    input_data = sys.stdin.read().strip().splitlines()
    idx = 0
    
    # Read the size of X and Y
    N = int(input_data[idx])
    idx += 1

    # Read the array X
    X = list(map(int, input_data[idx].split()))
    idx += 1

    # Read the array Y
    Y = list(map(int, input_data[idx].split()))
    idx += 1

    # Read the number of queries
    Q = int(input_data[idx])
    idx += 1

    # Read the queries
    T, B, L, R = [], [], [], []
    for _ in range(Q):
        t, b, l, r = map(int, input_data[idx].split())
        T.append(t)
        B.append(b)
        L.append(l)
        R.append(r)
        idx += 1

    # Call the mosaic function and print the results
    results = mosaic(X, Y, T, B, L, R)
    for result in results:
        print(result)

if __name__ == "__main__":
    main()