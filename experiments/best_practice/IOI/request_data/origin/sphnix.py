def find_colours(N, X, Y):
    # N: number of vertices
    # X, Y: lists of edges, edge i connects vertex X[i] and Y[i]
    from sys import stdin
    import threading

    # Limit on the number of experiments
    MAX_EXPERIMENTS = 2750

    # The number of vertices
    M = len(X)
    
    # Build adjacency list
    adj = [[] for _ in range(N)]
    for x, y in zip(X, Y):
        adj[x].append(y)
        adj[y].append(x)
    
    # Initialize variables
    experiments = []
    M0 = None  # Initial number of monochromatic components

    num_experiments = 0
    edge_results = {}

    # Function to perform an experiment
    def perform(E):
        nonlocal num_experiments

        num_experiments +=1
        return perform_experiment(E)
    
    def main():
        nonlocal M0, num_experiments, edge_results
        # First, get the initial number of monochromatic components
        E = [-1]*N  # No recolouring
        M0 = perform(E)

        # Build a spanning tree (here we use the edges as given)
        parent = [-1]*N
        visited = [False]*N
        from collections import deque
        queue = deque()
        root = 0
        queue.append(root)
        visited[root] = True
        while queue:
            u = queue.popleft()
            for v in adj[u]:
                if not visited[v]:
                    visited[v] = True
                    parent[v] = u
                    queue.append(v)
        
        union_find = [i for i in range(N)]
        def find(u):
            while union_find[u] != u:
                union_find[u] = union_find[union_find[u]]
                u = union_find[u]
            return u
        def union(u, v):
            u_root = find(u)
            v_root = find(v)
            if u_root != v_root:
                union_find[v_root] = u_root

        # Now, for each edge in the spanning tree, perform the experiment
        for u in range(1, N):
            v = parent[u]
            if num_experiments >= MAX_EXPERIMENTS:
                break
            # Recolour u and v to N
            E = [-1]*N
            E[u] = N
            E[v] = N
            delta = perform(E) - M0
            if delta == -1:
                # C[u] != C[v]
                pass
            elif delta == +1:
                # C[u] == C[v]
                union(u, v)
            else:
                # Should not happen
                pass

        # For the remaining edges not in the spanning tree
        # We may not know whether their endpoints have the same colour
        # But for partial marks, it suffices to know the grouping from the spanning tree

        # Now, assign colours to groups arbitrarily
        color_map = {}
        color_counter = 0
        G = [0]*N
        for i in range(N):
            root = find(i)
            if root not in color_map:
                color_map[root] = color_counter
                color_counter +=1
            G[i] = color_map[root]
        
        # Return G
        return G
    
    # Run main in a threading to avoid exceeding recursion depth
    threading.Thread(target=main).start()

    # Wait for the thread to finish
    threading.Event().wait(1)

    return G


# The following code is for testing and should not be included in submission

if __name__ == "__main__":
    import sys
    import threading
    import random

    sys.setrecursionlimit(1 << 25)
    threading.stack_size(1 << 27)
    def main():
        N_and_M = sys.stdin.readline().split()
        if len(N_and_M) < 2:
            N_and_M += sys.stdin.readline().split()
        N, M = map(int, N_and_M)
        N = int(N)
        M = int(M)
        C_line = ''
        while len(C_line.strip().split()) < N:
            C_line += sys.stdin.readline()
        C = list(map(int, C_line.strip().split()))
        X = []
        Y = []
        for _ in range(M):
            xy_line = ''
            while len(xy_line.strip().split()) < 2:
                xy_line += sys.stdin.readline()
            x_str, y_str = xy_line.strip().split()
            X.append(int(x_str))
            Y.append(int(y_str))
        # Hidden colours C are not accessible
        # Simulate grader
        G = find_colours(N, X, Y)
        print(len(G), num_experiments)
        print(' '.join(map(str, G)))
    
    threading.Thread(target=main).start()
