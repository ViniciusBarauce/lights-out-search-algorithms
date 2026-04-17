"""
Lights Out – Problema das Luzes
Estudo Dirigido 1 – Inteligência Artificial – UTP
Algoritmos: BFS, DFS, Greedy, A*, Hill Climbing, Simulated Annealing
"""

import time
import tracemalloc
import random
import heapq
import math
import csv
from collections import deque

# ══════════════════════════════════════════════
#  UTILITÁRIOS DO TABULEIRO
# ══════════════════════════════════════════════

def toggle(board, n, r, c):
    """Retorna novo tabuleiro após clicar em (r,c)."""
    b = [list(row) for row in board]
    for dr, dc in [(0,0),(-1,0),(1,0),(0,-1),(0,1)]:
        nr, nc = r+dr, c+dc
        if 0 <= nr < n and 0 <= nc < n:
            b[nr][nc] ^= 1
    return tuple(tuple(row) for row in b)

def goal_state(n):
    return tuple(tuple(1 for _ in range(n)) for _ in range(n))

def is_goal(board):
    return all(cell == 1 for row in board for cell in row)

def heuristic(board, n):
    """Número de luzes apagadas (admissível)."""
    return sum(1 for row in board for cell in row if cell == 0)

def random_board(n, seed=None):
    """Gera tabuleiro garantidamente solucionável partindo do objetivo."""
    if seed is not None:
        random.seed(seed)
    b = goal_state(n)
    moves = random.randint(n, n * n)
    for _ in range(moves):
        r, c = random.randint(0, n-1), random.randint(0, n-1)
        b = toggle(b, n, r, c)
    return b

def board_str(board):
    return '\n'.join(' '.join('●' if c else '○' for c in row) for row in board)

# ══════════════════════════════════════════════
#  LIMITES
# ══════════════════════════════════════════════

MEMORY_LIMIT = 150_000   # estados máximos antes de desistir
TIME_LIMIT   = 20        # segundos por execução

# ══════════════════════════════════════════════
#  BFS – Busca em Largura
# ══════════════════════════════════════════════

def bfs(start, n, time_limit=TIME_LIMIT):
    if is_goal(start):
        return 0, 0.0, 1
    t0 = time.time()
    frontier = deque([(start, 0)])
    visited = {start}
    while frontier:
        if time.time() - t0 > time_limit:
            return None, None, len(visited)
        if len(visited) > MEMORY_LIMIT:
            return None, None, len(visited)
        state, depth = frontier.popleft()
        for r in range(n):
            for c in range(n):
                ns = toggle(state, n, r, c)
                if ns not in visited:
                    if is_goal(ns):
                        return depth + 1, time.time() - t0, len(visited)
                    visited.add(ns)
                    frontier.append((ns, depth + 1))
    return None, time.time() - t0, len(visited)

# ══════════════════════════════════════════════
#  DFS – Busca em Profundidade
# ══════════════════════════════════════════════

def dfs(start, n, time_limit=TIME_LIMIT):
    if is_goal(start):
        return 0, 0.0, 1
    depth_limit = n * n
    t0 = time.time()
    stack = [(start, 0, frozenset([start]))]
    nodes = 1
    while stack:
        if time.time() - t0 > time_limit:
            return None, None, nodes
        if nodes > MEMORY_LIMIT:
            return None, None, nodes
        state, depth, path = stack.pop()
        if depth >= depth_limit:
            continue
        for r in range(n):
            for c in range(n):
                ns = toggle(state, n, r, c)
                if ns not in path:
                    nodes += 1
                    if is_goal(ns):
                        return depth + 1, time.time() - t0, nodes
                    stack.append((ns, depth + 1, path | {ns}))
    return None, time.time() - t0, nodes

# ══════════════════════════════════════════════
#  GREEDY – Busca Gulosa
# ══════════════════════════════════════════════

def greedy(start, n, time_limit=TIME_LIMIT):
    if is_goal(start):
        return 0, 0.0, 1
    t0 = time.time()
    frontier = [(heuristic(start, n), start, 0)]
    visited = {start}
    while frontier:
        if time.time() - t0 > time_limit:
            return None, None, len(visited)
        if len(visited) > MEMORY_LIMIT:
            return None, None, len(visited)
        _, state, depth = heapq.heappop(frontier)
        for r in range(n):
            for c in range(n):
                ns = toggle(state, n, r, c)
                if ns not in visited:
                    if is_goal(ns):
                        return depth + 1, time.time() - t0, len(visited)
                    visited.add(ns)
                    heapq.heappush(frontier, (heuristic(ns, n), ns, depth + 1))
    return None, time.time() - t0, len(visited)

# ══════════════════════════════════════════════
#  A* – Algoritmo A*
# ══════════════════════════════════════════════

def astar(start, n, time_limit=TIME_LIMIT):
    if is_goal(start):
        return 0, 0.0, 1
    t0 = time.time()
    counter = 0
    frontier = [(heuristic(start, n), counter, start, 0)]
    dist = {start: 0}
    visited = 0
    while frontier:
        if time.time() - t0 > time_limit:
            return None, None, visited
        if visited > MEMORY_LIMIT:
            return None, None, visited
        _, _, state, g = heapq.heappop(frontier)
        visited += 1
        if is_goal(state):
            return g, time.time() - t0, visited
        for r in range(n):
            for c in range(n):
                ns = toggle(state, n, r, c)
                ng = g + 1
                if ng < dist.get(ns, float('inf')):
                    dist[ns] = ng
                    counter += 1
                    heapq.heappush(frontier,
                                   (ng + heuristic(ns, n), counter, ns, ng))
    return None, time.time() - t0, visited

# ══════════════════════════════════════════════
#  HILL CLIMBING – com Reinícios Aleatórios
# ══════════════════════════════════════════════

def hill_climbing(start, n, time_limit=TIME_LIMIT, restarts=30):
    t0 = time.time()
    evaluations = 0

    def climb(state):
        nonlocal evaluations
        current, current_h = state, heuristic(state, n)
        path_len = 0
        while True:
            if time.time() - t0 > time_limit:
                return None
            if is_goal(current):
                return path_len
            best_h, best_ns = current_h, None
            for r in range(n):
                for c in range(n):
                    ns = toggle(current, n, r, c)
                    h = heuristic(ns, n)
                    evaluations += 1
                    if h < best_h:
                        best_h, best_ns = h, ns
            if best_ns is None:
                return None        # mínimo local
            current, current_h = best_ns, best_h
            path_len += 1

    for _ in range(restarts):
        if time.time() - t0 > time_limit:
            break
        candidate = start
        for _ in range(random.randint(0, n)):
            r, c = random.randint(0, n-1), random.randint(0, n-1)
            candidate = toggle(candidate, n, r, c)
        result = climb(candidate)
        if result is not None:
            return result, time.time() - t0, evaluations

    return None, time.time() - t0, evaluations

# ══════════════════════════════════════════════
#  SIMULATED ANNEALING
# ══════════════════════════════════════════════

def simulated_annealing(start, n, time_limit=TIME_LIMIT, max_iter=500_000):
    t0 = time.time()
    current = start
    current_h = heuristic(current, n)
    T = 2.0
    alpha = 1 - 3e-5
    steps = 0
    for i in range(max_iter):
        if time.time() - t0 > time_limit:
            break
        if is_goal(current):
            return steps, time.time() - t0, i
        r, c = random.randint(0, n-1), random.randint(0, n-1)
        ns = toggle(current, n, r, c)
        nh = heuristic(ns, n)
        delta = nh - current_h
        if delta < 0 or (T > 1e-9 and random.random() < math.exp(-delta / T)):
            current, current_h = ns, nh
            steps += 1
        T *= alpha
    found = steps if is_goal(current) else None
    return found, time.time() - t0, max_iter

# ══════════════════════════════════════════════
#  RUNNER DE EXPERIMENTO
# ══════════════════════════════════════════════

def run(n, board, name, fn):
    tracemalloc.start()
    result = fn(board, n)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    steps, t, nodes = result
    return {
        'algorithm': name,
        'n': n,
        'steps': steps,
        'time_s': round(t, 4) if t is not None else None,
        'memory_kb': round(peak / 1024, 2),
        'nodes': nodes,
    }

def fmt(v):
    return str(v) if v is not None else 'N/A'

# ══════════════════════════════════════════════
#  CONFIGURAÇÃO DOS EXPERIMENTOS
# ══════════════════════════════════════════════

SIZES = [2, 3, 5, 7, 10]

ALGOS = [
    ('BFS',                bfs),
    ('DFS',                dfs),
    ('Greedy',             greedy),
    ('A*',                 astar),
    ('Hill Climbing',      hill_climbing),
    ('Simulated Annealing',simulated_annealing),
]

# Limites: tamanho máximo (inclusive) para cada algoritmo
MAX_SIZE = {
    'BFS':   3,
    'DFS':   3,
    'Greedy':5,
    'A*':    5,
    'Hill Climbing':      10,
    'Simulated Annealing':10,
}

# ══════════════════════════════════════════════
#  EXECUÇÃO
# ══════════════════════════════════════════════

random.seed(42)
all_results = []

print("=" * 72)
print("        LIGHTS OUT – Análise Comparativa de Algoritmos de Busca")
print("=" * 72)

for n in SIZES:
    board = random_board(n)
    print(f"\n╔═══ Tabuleiro {n}×{n} ══════════════════════")
    for line in board_str(board).split('\n'):
        print(f"║  {line}")
    print(f"╚{'═'*40}")

    for name, fn in ALGOS:
        if n > MAX_SIZE[name]:
            print(f"  {name:<25} → ignorado (instância grande demais)")
            all_results.append({
                'algorithm': name, 'n': n,
                'steps': None, 'time_s': None,
                'memory_kb': None, 'nodes': None,
            })
            continue

        print(f"  {name:<25} … ", end='', flush=True)
        res = run(n, board, name, fn)
        all_results.append(res)
        s = fmt(res['steps'])
        t = fmt(res['time_s'])
        m = fmt(res['memory_kb'])
        k = fmt(res['nodes'])
        print(f"passos={s:>5}  t={t:>8}s  mem={m:>9}KB  nós={k}")

# ══════════════════════════════════════════════
#  TABELA RESUMO
# ══════════════════════════════════════════════

print("\n\n" + "=" * 80)
print("                              TABELA RESUMO FINAL")
print("=" * 80)
print(f"{'Algoritmo':<26} {'N':>3} {'Passos':>8} {'Tempo(s)':>10} {'Mem(KB)':>10} {'Nós':>10}")
print('-' * 80)
for r in all_results:
    print(f"{r['algorithm']:<26} {r['n']:>3} "
          f"{fmt(r['steps']):>8} {fmt(r['time_s']):>10} "
          f"{fmt(r['memory_kb']):>10} {fmt(r['nodes']):>10}")

# Salva CSV
with open('/home/claude/resultados.csv', 'w', newline='', encoding='utf-8') as f:
    w = csv.DictWriter(f, fieldnames=['algorithm','n','steps','time_s','memory_kb','nodes'])
    w.writeheader()
    w.writerows(all_results)

print("\nCSV salvo: resultados.csv")
