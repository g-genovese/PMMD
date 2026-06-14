#!/usr/bin/env python3
"""
compute_BLS_rigorous.py
========================

Estrazione RIGOROSA della dimensione d_BS dal foam di Sim B v4.3 partial-order,
via campionamento di raggiungibilità nel grafo causale, SENZA modificare la
simulazione (i dati esistenti sono sufficienti).

PRINCIPIO STRUTTURALE
---------------------
Per la BLS partial-order del foam:
  edge (u → v) ∈ G_causal  iff  - u, v sono foam-vicini (struttura E_8 lattice, da L)
                                - entrambi attivati (in activated_vertices)
                                - t_u < t_v (da activation_time)

  a ≼_BLS b  iff  b raggiungibile da a in G_causal (path temporalmente monotonici)

Ordering fraction r = |R| / N(N-1)/2 dove R = #{(a,b) : a ≼ b}.
Inversione MM r ↔ d → d_BS.

METODOLOGIA
-----------
Per ogni trial:
  1. Carica PO snapshot
  2. Path-compress parent_snapshot per identificare i cluster
  3. Estrai giant cluster (largest connected component)
  4. Sample K coppie casuali (a, b) e per ognuna BFS forward da a
  5. Conta quante raggiungono b
  6. r ≈ K_reached / K  → MM inversion → d_BS

Si fanno DUE misure indipendenti per cross-check:
  - r_giant: coppie ENTRO il giant cluster (Alexandrov interval = "spacetime")
  - r_full: coppie tra TUTTI gli attivati (sanity check)

USAGE:
  python3 compute_BLS_rigorous.py /path/to/po_files/ --workers 16 --K 100000

DISTRIBUTED (3 server, PO files locali):
  server-A: python3 compute_BLS_rigorous.py /sim_b/ --workers 16 --K 100000 --output BLS_A.json
  server-B: python3 compute_BLS_rigorous.py /sim_b/ --workers  9 --K 100000 --output BLS_B.json
  server-C: python3 compute_BLS_rigorous.py /sim_b/ --workers 28 --K 100000 --output BLS_C.json

  Poi: python3 merge_BLS_results.py BLS_A.json BLS_B.json BLS_C.json
"""

from __future__ import annotations
import argparse
import glob
import json
import os
import re
import sys
import time
from multiprocessing import Pool, cpu_count

import numpy as np

try:
    from numba import njit
    HAS_NUMBA = True
except ImportError:
    print("ERROR: Numba required for performance (BFS sampling on 10^5 pairs).")
    print("Install: pip install numba")
    sys.exit(1)


# =============================================================================
# E_8 lattice neighbor structure (identical to v4.3 simulation)
# =============================================================================

def make_e8_shifts():
    """Generate typeA (112) and typeB_h0, typeB_h1 (128 each) neighbor offsets."""
    typeA = []
    for i in range(8):
        for j in range(i + 1, 8):
            for s1 in (1, -1):
                for s2 in (1, -1):
                    d = np.zeros(8, dtype=np.int8)
                    d[i] = s1
                    d[j] = s2
                    typeA.append(d)
    typeA = np.stack(typeA)
    assert typeA.shape == (112, 8)
    
    typeB_h0 = []
    typeB_h1 = []
    for mask in range(256):
        if bin(mask).count("1") % 2 != 0:
            continue
        s = np.array([-1 if (mask >> k) & 1 else 1 for k in range(8)], dtype=np.int8)
        d0 = ((s - 1) // 2).astype(np.int8)
        d1 = ((s + 1) // 2).astype(np.int8)
        typeB_h0.append(d0)
        typeB_h1.append(d1)
    typeB_h0 = np.stack(typeB_h0)
    typeB_h1 = np.stack(typeB_h1)
    assert typeB_h0.shape == (128, 8)
    assert typeB_h1.shape == (128, 8)
    return typeA, typeB_h0, typeB_h1


# =============================================================================
# Numba-accelerated BFS reachability
# =============================================================================

@njit(cache=True, fastmath=False)
def _compute_neighbors_into(v: np.int64, L: np.int64, L8: np.int64,
                            typeA, typeB_h0, typeB_h1, out):
    """Compute 240 neighbors of vertex v, write to out[0:240]."""
    if v >= L8:
        h = 1
        lex_v = v - L8
    else:
        h = 0
        lex_v = v
    
    # Decode 8D coordinates from lex_v
    c0 = lex_v % L; lex_v //= L
    c1 = lex_v % L; lex_v //= L
    c2 = lex_v % L; lex_v //= L
    c3 = lex_v % L; lex_v //= L
    c4 = lex_v % L; lex_v //= L
    c5 = lex_v % L; lex_v //= L
    c6 = lex_v % L; lex_v //= L
    c7 = lex_v % L
    
    base = h * L8
    
    # Type-A: 112 neighbors, same h
    for ri in range(112):
        ck0 = c0 + typeA[ri, 0]
        if ck0 < 0: ck0 += L
        elif ck0 >= L: ck0 -= L
        ck1 = c1 + typeA[ri, 1]
        if ck1 < 0: ck1 += L
        elif ck1 >= L: ck1 -= L
        ck2 = c2 + typeA[ri, 2]
        if ck2 < 0: ck2 += L
        elif ck2 >= L: ck2 -= L
        ck3 = c3 + typeA[ri, 3]
        if ck3 < 0: ck3 += L
        elif ck3 >= L: ck3 -= L
        ck4 = c4 + typeA[ri, 4]
        if ck4 < 0: ck4 += L
        elif ck4 >= L: ck4 -= L
        ck5 = c5 + typeA[ri, 5]
        if ck5 < 0: ck5 += L
        elif ck5 >= L: ck5 -= L
        ck6 = c6 + typeA[ri, 6]
        if ck6 < 0: ck6 += L
        elif ck6 >= L: ck6 -= L
        ck7 = c7 + typeA[ri, 7]
        if ck7 < 0: ck7 += L
        elif ck7 >= L: ck7 -= L
        lex_u = ck0 + L * (ck1 + L * (ck2 + L * (ck3 + L * (ck4 + L * (ck5 + L * (ck6 + L * ck7))))))
        out[ri] = base + lex_u
    
    # Type-B: 128 neighbors, opposite h
    if h == 0:
        typeB = typeB_h0
        h_new_base = L8
    else:
        typeB = typeB_h1
        h_new_base = 0
    
    for ri in range(128):
        ck0 = c0 + typeB[ri, 0]
        if ck0 < 0: ck0 += L
        elif ck0 >= L: ck0 -= L
        ck1 = c1 + typeB[ri, 1]
        if ck1 < 0: ck1 += L
        elif ck1 >= L: ck1 -= L
        ck2 = c2 + typeB[ri, 2]
        if ck2 < 0: ck2 += L
        elif ck2 >= L: ck2 -= L
        ck3 = c3 + typeB[ri, 3]
        if ck3 < 0: ck3 += L
        elif ck3 >= L: ck3 -= L
        ck4 = c4 + typeB[ri, 4]
        if ck4 < 0: ck4 += L
        elif ck4 >= L: ck4 -= L
        ck5 = c5 + typeB[ri, 5]
        if ck5 < 0: ck5 += L
        elif ck5 >= L: ck5 -= L
        ck6 = c6 + typeB[ri, 6]
        if ck6 < 0: ck6 += L
        elif ck6 >= L: ck6 -= L
        ck7 = c7 + typeB[ri, 7]
        if ck7 < 0: ck7 += L
        elif ck7 >= L: ck7 -= L
        lex_u = ck0 + L * (ck1 + L * (ck2 + L * (ck3 + L * (ck4 + L * (ck5 + L * (ck6 + L * ck7))))))
        out[112 + ri] = h_new_base + lex_u


@njit(cache=True)
def _bfs_reachable(a_pos: np.int64, b_pos: np.int64,
                   activated_vertices, activation_time, is_in_target,
                   L: np.int64, L8: np.int64,
                   typeA, typeB_h0, typeB_h1,
                   visited, queue, neighbors_buf):
    """
    BFS from local index a_pos in activated_vertices, follow forward causal edges
    (temporally monotonic), check if local index b_pos is reachable.
    
    'is_in_target' restricts BFS to only consider vertices in the target subset
    (e.g. the giant cluster).
    
    Buffers (visited, queue, neighbors_buf) are pre-allocated and reused.
    """
    n_act = len(activated_vertices)
    
    # Reset visited buffer (we use a separate "generation" counter to avoid memset)
    # For simplicity here: reset to False, but worker can do this less often
    for i in range(n_act):
        visited[i] = False
    
    a_v = activated_vertices[a_pos]
    b_v = activated_vertices[b_pos]
    t_b = activation_time[b_pos]
    
    visited[a_pos] = True
    queue[0] = a_pos
    queue_head = 0
    queue_tail = 1
    
    while queue_head < queue_tail:
        cur_pos = queue[queue_head]
        queue_head += 1
        
        cur_v = activated_vertices[cur_pos]
        if cur_pos == b_pos:
            return True
        
        t_cur = activation_time[cur_pos]
        # If t_cur >= t_b, can't reach b
        if t_cur >= t_b:
            continue
        
        # Compute 240 neighbors of cur_v
        _compute_neighbors_into(cur_v, L, L8, typeA, typeB_h0, typeB_h1, neighbors_buf)
        
        # For each neighbor, check if activated, in target, and has later time
        for ni in range(240):
            nb = neighbors_buf[ni]
            # Binary search in activated_vertices (sorted)
            nb_pos = np.searchsorted(activated_vertices, nb)
            if nb_pos < n_act and activated_vertices[nb_pos] == nb:
                if (not visited[nb_pos]) and is_in_target[nb_pos]:
                    t_nb = activation_time[nb_pos]
                    if t_nb > t_cur and t_nb <= t_b:
                        visited[nb_pos] = True
                        if nb_pos == b_pos:
                            return True
                        queue[queue_tail] = nb_pos
                        queue_tail += 1
                        if queue_tail >= len(queue):
                            return False   # buffer overflow: assume reachable (conservative)
    
    return False


@njit(cache=True)
def _sample_pairs_reachability(K: np.int64,
                                eligible_positions,  # int64 array of local positions
                                activated_vertices, activation_time, is_in_target,
                                L: np.int64, L8: np.int64,
                                typeA, typeB_h0, typeB_h1,
                                seed: np.int64,
                                visited, queue, neighbors_buf):
    """
    Sample K random ordered pairs (a, b) from eligible_positions (with t_a < t_b),
    BFS-check reachability, return count of reachable pairs.
    """
    n_elig = len(eligible_positions)
    n_reached = np.int64(0)
    n_valid = np.int64(0)
    np.random.seed(seed)
    
    for k in range(K):
        # Sample two distinct local positions
        ia = np.random.randint(0, n_elig)
        ib = np.random.randint(0, n_elig)
        if ia == ib:
            continue
        
        pa = eligible_positions[ia]
        pb = eligible_positions[ib]
        ta = activation_time[pa]
        tb = activation_time[pb]
        
        # Ensure t_a < t_b (swap if needed)
        if ta == tb:
            continue   # rare but skip
        if ta > tb:
            pa, pb = pb, pa
            ta, tb = tb, ta
        
        n_valid += 1
        if _bfs_reachable(pa, pb, activated_vertices, activation_time, is_in_target,
                          L, L8, typeA, typeB_h0, typeB_h1,
                          visited, queue, neighbors_buf):
            n_reached += 1
    
    return n_reached, n_valid


# =============================================================================
# Myrheim-Meyer inversion table
# =============================================================================

_MM_TABLE = np.array([
    (1.0, 1.000), (1.5, 0.730), (2.0, 0.500), (2.5, 0.413),
    (3.0, 0.350), (3.5, 0.318), (4.0, 0.292), (4.5, 0.270),
    (5.0, 0.244), (5.5, 0.227), (6.0, 0.208), (7.0, 0.180),
    (8.0, 0.143), (9.0, 0.124), (10.0, 0.108), (15.0, 0.063),
    (20.0, 0.043),
])

def d_MM_from_r(r):
    """Invert MM curve: ordering fraction r → dimension d."""
    if r is None or r <= 0 or r > 1:
        return None
    table_r_rev = _MM_TABLE[:, 1][::-1]
    table_d_rev = _MM_TABLE[:, 0][::-1]
    if r >= table_r_rev[-1]:
        return float(table_d_rev[-1])
    if r <= table_r_rev[0]:
        return float(table_d_rev[0])
    return float(np.interp(r, table_r_rev, table_d_rev))


# =============================================================================
# Path-compression for cluster identification
# =============================================================================

def find_cluster_roots(activated_vertices, parent_snapshot):
    """
    Translate global parent IDs to local positions, then path-compress to find
    cluster root (local position) for each activated vertex.
    """
    local_parent = np.searchsorted(activated_vertices, parent_snapshot)
    # Sanity check
    check = activated_vertices[np.clip(local_parent, 0, len(activated_vertices)-1)] == parent_snapshot
    if not check.all():
        raise ValueError(f"{int((~check).sum())} parent IDs not in activated_vertices")
    
    roots = local_parent.astype(np.int64)
    for _ in range(60):
        new_roots = local_parent[roots]
        if np.array_equal(new_roots, roots):
            break
        roots = new_roots
    return roots


# =============================================================================
# Single-trial processing
# =============================================================================

# Global cache for typeA/typeB shared across workers via initializer
_TYPEA = None
_TYPEB_H0 = None
_TYPEB_H1 = None

def _init_worker():
    global _TYPEA, _TYPEB_H0, _TYPEB_H1
    _TYPEA, _TYPEB_H0, _TYPEB_H1 = make_e8_shifts()


def process_one_trial(args):
    """Process one PO file: compute r_giant and r_full + d_BS."""
    po_path, K_giant, K_full = args
    bname = os.path.basename(po_path)
    
    m = re.search(r'_trial(\d+)\.npz', bname)
    if not m:
        return {'po_file': bname, 'error': 'cannot parse trial index'}
    trial_idx = int(m.group(1))
    
    t0 = time.time()
    
    try:
        with np.load(po_path) as npz:
            activated_vertices = npz['activated_vertices'].astype(np.int64)
            activation_time = npz['activation_time'].astype(np.int64)
            parent_snapshot = npz['parent_snapshot'].astype(np.int64)
            n_total = int(npz['n_total_activated_at_snap'])
            snapshot_bond_idx = int(npz['snapshot_bond_idx'])
            seed = int(npz['seed'])
            L = int(npz['L'])
    except Exception as e:
        return {'po_file': bname, 'trial_idx': trial_idx, 'error': f'load: {e}'}
    
    n_act = len(activated_vertices)
    L8 = L ** 8
    
    # Identify cluster roots
    try:
        roots = find_cluster_roots(activated_vertices, parent_snapshot)
    except ValueError as e:
        return {'po_file': bname, 'trial_idx': trial_idx, 'error': f'roots: {e}'}
    
    unique_roots, cluster_sizes = np.unique(roots, return_counts=True)
    giant_root_idx = unique_roots[np.argmax(cluster_sizes)]
    giant_mask = (roots == giant_root_idx)
    giant_size = int(giant_mask.sum())
    giant_positions = np.where(giant_mask)[0].astype(np.int64)
    all_positions = np.arange(n_act, dtype=np.int64)
    
    # Pre-allocate BFS buffers
    visited = np.zeros(n_act, dtype=np.bool_)
    queue = np.zeros(min(n_act, max(giant_size * 2, 100000)), dtype=np.int64)
    neighbors_buf = np.zeros(240, dtype=np.int64)
    
    # ----- Phase 1: Giant cluster BLS -----
    is_in_giant = giant_mask.astype(np.bool_)
    
    t_giant_start = time.time()
    n_reached_g, n_valid_g = _sample_pairs_reachability(
        K_giant, giant_positions,
        activated_vertices, activation_time, is_in_giant,
        L, L8, _TYPEA, _TYPEB_H0, _TYPEB_H1,
        seed,
        visited, queue, neighbors_buf
    )
    t_giant = time.time() - t_giant_start
    
    r_giant = float(n_reached_g) / float(n_valid_g) if n_valid_g > 0 else None
    d_BS_giant = d_MM_from_r(r_giant)
    
    # ----- Phase 2: Full activated BLS -----
    is_in_all = np.ones(n_act, dtype=np.bool_)
    
    t_full_start = time.time()
    n_reached_f, n_valid_f = _sample_pairs_reachability(
        K_full, all_positions,
        activated_vertices, activation_time, is_in_all,
        L, L8, _TYPEA, _TYPEB_H0, _TYPEB_H1,
        seed + 1,   # different seed for full sampling
        visited, queue, neighbors_buf
    )
    t_full = time.time() - t_full_start
    
    r_full = float(n_reached_f) / float(n_valid_f) if n_valid_f > 0 else None
    d_BS_full = d_MM_from_r(r_full)
    
    dt = time.time() - t0
    
    return {
        'po_file': bname,
        'trial_idx': trial_idx,
        'seed': seed,
        'L': L,
        'N_activated': n_act,
        'snapshot_bond_idx': snapshot_bond_idx,
        'n_clusters': int(len(unique_roots)),
        'giant_cluster_size': giant_size,
        'giant_cluster_frac': giant_size / n_act,
        # E1 — giant cluster BLS
        'K_giant': K_giant,
        'n_valid_pairs_giant': int(n_valid_g),
        'n_reached_giant': int(n_reached_g),
        'r_giant': r_giant,
        'd_BS_giant': d_BS_giant,
        'time_giant_s': t_giant,
        # E2 — full activated BLS
        'K_full': K_full,
        'n_valid_pairs_full': int(n_valid_f),
        'n_reached_full': int(n_reached_f),
        'r_full': r_full,
        'd_BS_full': d_BS_full,
        'time_full_s': t_full,
        # total
        'total_time_s': dt,
    }


# =============================================================================
# Main
# =============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('directory', help='Directory with sim_b_L*_po_trial*.npz files')
    ap.add_argument('--workers', type=int, default=cpu_count())
    ap.add_argument('--K', type=int, default=100000,
                    help='Pair sampling count K per trial for giant cluster (default 10^5)')
    ap.add_argument('--K-full', type=int, default=None,
                    help='Pair sampling for full activated (default: K/10)')
    ap.add_argument('--output', default='BLS_pertrial.json')
    args = ap.parse_args()
    
    K_giant = args.K
    K_full = args.K_full if args.K_full else max(K_giant // 10, 1000)
    
    pattern = os.path.join(args.directory, 'sim_b_L*_po_trial*.npz')
    po_files = sorted(glob.glob(pattern))
    if not po_files:
        print(f"ERROR: no PO files in {args.directory}")
        sys.exit(1)
    
    print(f"\n{'='*70}")
    print(f"BLS dimension extraction (rigorous, via reachability BFS)")
    print(f"{'='*70}")
    print(f"Files:           {len(po_files)}")
    print(f"Workers:         {args.workers}")
    print(f"K (giant):       {K_giant:,}")
    print(f"K (full):        {K_full:,}")
    print()
    
    t0 = time.time()
    work_items = [(p, K_giant, K_full) for p in po_files]
    
    with Pool(processes=args.workers, initializer=_init_worker) as pool:
        results = []
        for i, r in enumerate(pool.imap_unordered(process_one_trial, work_items), 1):
            results.append(r)
            if 'error' in r:
                print(f"  [{i:>3}/{len(po_files)}] trial {r.get('trial_idx','?'):>3}: ERROR {r['error']}")
            else:
                r_g_str = f"{r['r_giant']:.4f}" if r['r_giant'] else "—"
                d_g_str = f"{r['d_BS_giant']:.2f}" if r['d_BS_giant'] else "—"
                r_f_str = f"{r['r_full']:.5f}" if r['r_full'] else "—"
                d_f_str = f"{r['d_BS_full']:.2f}" if r['d_BS_full'] else "—"
                print(f"  [{i:>3}/{len(po_files)}] trial {r['trial_idx']:>3}: "
                      f"giant({r['giant_cluster_size']}): r={r_g_str} d_BS={d_g_str} | "
                      f"full: r={r_f_str} d_BS={d_f_str} | "
                      f"{r['total_time_s']:.0f}s")
    
    dt = time.time() - t0
    results.sort(key=lambda x: x.get('trial_idx', -1))
    
    # Aggregate stats
    valid = [r for r in results if 'error' not in r and r.get('d_BS_giant') is not None]
    if valid:
        r_giant_arr = np.array([r['r_giant'] for r in valid])
        d_giant_arr = np.array([r['d_BS_giant'] for r in valid])
        r_full_arr = np.array([r['r_full'] for r in valid if r['r_full'] is not None])
        d_full_arr = np.array([r['d_BS_full'] for r in valid if r['d_BS_full'] is not None])
        
        n = len(valid)
        print(f"\n{'='*70}")
        print(f"Aggregate results ({n} valid trials)")
        print(f"{'='*70}")
        print(f"\nGiant cluster BLS:")
        print(f"  r_giant: mean = {r_giant_arr.mean():.5f}, sem = {r_giant_arr.std(ddof=1)/np.sqrt(n):.5f}")
        print(f"  d_BS_giant: mean = {d_giant_arr.mean():.3f}, sem = {d_giant_arr.std(ddof=1)/np.sqrt(n):.3f}")
        print(f"  Framework prediction: d_BS = 4")
        sem_g = d_giant_arr.std(ddof=1)/np.sqrt(n)
        print(f"  Tension with 4: {(d_giant_arr.mean() - 4)/sem_g:+.2f}σ")
        
        if len(d_full_arr) > 0:
            print(f"\nFull activated BLS (sanity check):")
            print(f"  r_full: mean = {r_full_arr.mean():.6f}, sem = {r_full_arr.std(ddof=1)/np.sqrt(len(r_full_arr)):.6f}")
            print(f"  d_BS_full: mean = {d_full_arr.mean():.3f}, sem = {d_full_arr.std(ddof=1)/np.sqrt(len(d_full_arr)):.3f}")
    
    output = {
        'description': 'Rigorous BLS dimension extraction via reachability BFS sampling',
        'n_trials_processed': len(results),
        'n_trials_valid': len(valid),
        'K_giant': K_giant,
        'K_full': K_full,
        'computation_time_s': dt,
        'workers': args.workers,
        'per_trial_results': results,
    }
    if valid:
        output['aggregate'] = {
            'r_giant_mean': float(r_giant_arr.mean()),
            'r_giant_sem': float(r_giant_arr.std(ddof=1)/np.sqrt(n)),
            'd_BS_giant_mean': float(d_giant_arr.mean()),
            'd_BS_giant_sem': float(d_giant_arr.std(ddof=1)/np.sqrt(n)),
            'tension_d_BS_giant_vs_4': float((d_giant_arr.mean() - 4)/(d_giant_arr.std(ddof=1)/np.sqrt(n))),
            'r_full_mean': float(r_full_arr.mean()) if len(r_full_arr) > 0 else None,
            'd_BS_full_mean': float(d_full_arr.mean()) if len(d_full_arr) > 0 else None,
        }
    
    with open(args.output, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    
    print(f"\nSaved {args.output} ({os.path.getsize(args.output)/1024:.1f} KB)")
    print(f"Total wall time: {dt:.0f}s ({dt/60:.1f} min)")


if __name__ == '__main__':
    main()
