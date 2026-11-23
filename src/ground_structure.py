import numpy as np
import math
import matplotlib.pyplot as plt


def plot_truss(nodes, members, *, ax=None, show_nodes=True,
               node_size=40, member_color="k",
               node_color="red", lw=1.5, equal_aspect=True):

    if ax is None:
        fig, ax = plt.subplots()

    # Plot members
    for i, j in members:
        x = [nodes[i][0], nodes[j][0]]
        y = [nodes[i][1], nodes[j][1]]
        ax.plot(x, y, member_color, lw=lw)

    # Plot nodes
    if show_nodes:
        xs = [n[0] for n in nodes]
        ys = [n[1] for n in nodes]
        ax.scatter(xs, ys, c=node_color, s=node_size, zorder=3)

    if equal_aspect:
        ax.set_aspect("equal", adjustable="box")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.grid(True, linestyle="--", alpha=0.4)

    return ax



def generate_nodes(grid):
  origin = np.array([0, 0])
  nodes = np.zeros(((grid[0]+1) * (grid[1]+1), 2))
  nodes[0] = origin
  for r in range(grid[0]+1):
    for c in range(grid[1]+1):
      nodes[r * (grid[1]+1) + c] = origin + np.array([r, c])
  return nodes



def slope(A, B, tol=1e-12):
    A = np.asarray(A)
    B = np.asarray(B)

    dx = B[0] - A[0]
    dy = B[1] - A[1]

    return np.inf if abs(dx) < tol else dy / dx



def hypotenuse_set(n):
    hyp = set()

    for a in range(1, n):
        for b in range(a + 1, n):
            c2 = a*a + b*b
            c = int(math.isqrt(c2))   # safer than sqrt
            if c*c == c2 and c < n:
                hyp.add(c)
    return sorted(hyp)




def dist(node1, node2):
  return np.linalg.norm(node1 - node2)




def generate_connections(grid, nodes):

  conns = []
  Lmax = int(np.round(np.sqrt(grid[0]**2 + grid[1]**2)))
  combos = math.comb(len(nodes), 2)
  hypos = hypotenuse_set(Lmax)
  for i in range(len(nodes)):
    for j in range(len(nodes)):
      m = slope(nodes[i], nodes[j])
      if i < j and (dist(nodes[i], nodes[j]) % 1 != 0 or ((int(np.round(dist(nodes[i], nodes[j]))) in hypos) and (m != 0 and m != np.inf)) or dist(nodes[i], nodes[j]) == 1):
        conns.append(np.array([i, j]))
  conns = np.array(conns)
  return conns




def generate_ground_structure(grid, plot=False):
  n = generate_nodes(grid)
  c = generate_connections(grid, n)
  if plot == True:
    plot_truss(n, c)
  return n, c




def get_all_bars(nodes):
    all_bars = []
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            all_bars.append([i, j])
    return np.array(all_bars)




def connectivity_to_state(active_bars, nodes):
    all_bars = get_all_bars(nodes)
    state = np.zeros(len(all_bars), dtype=int)

    active_set = {tuple(b) for b in active_bars}

    for i, bar in enumerate(map(tuple, all_bars)):
        if bar in active_set:
            state[i] = 1

    return state




def state_to_connectivity(state, nodes):
    all_bars = get_all_bars(nodes)
    return all_bars[state == 1]




def state_to_index(state):
    return int("".join(map(str, state)), 2)




def index_to_state(index: int, n_bits: int) -> np.ndarray:
    return np.array(list(map(int, format(index, f'0{n_bits}b'))))




def cal_vol(nodes, state, A):  # A is in cm2
    volume = 0.0
    bars = state_to_connectivity(state, nodes)
    for (i, j) in bars:
        Li = np.linalg.norm(nodes[j] - nodes[i])
        volume += A * Li * 1e-4   # unit conversion if intended
    return volume  # in m3