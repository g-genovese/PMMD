#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
farm.py  --  one-console orchestrator for the PMMD foam-rigidity (induced-gravity) farm.

RUN IT ON THE WINDOWS WORKSTATION (cmd / PowerShell):
    py -m pip install paramiko
    py farm.py probe        # detect sockets / physical cores / free RAM on each server
    py farm.py setup        # deploy worker + venv + numpy/scipy
    py farm.py gen          # build the (N,seed) work list, partition it by (auto-tuned) cores
    py farm.py run          # launch detached, RAM-aware, NUMA-bound worker pools
    py farm.py status       # progress table
    py farm.py collect      # pull result JSONs to .\collected\<run>\
    py farm.py aggregate    # finite-size scaling -> G_foam(inf), compare to 1/(4 phi^10)
    py farm.py stop         # kill everything

Large-N (distributed Laplacian over the 10Gbps subnet):
    py farm.py run-mpi --N 4000000 --samples 8 --fast-iface 10.0.0.0/24

AUTO-TUNING: set "workers":"auto" / "numa_nodes":"auto" below and the farm sizes itself to each
server: workers = min(physical cores, RAM-budget), NUMA binding from the detected socket count,
always leaving a reserve for the OS (max(RESERVE_FRAC*RAM, RESERVE_GB)). The per-worker memory
budget is estimated for the largest N and re-checked LIVE on each server at launch.
Notebooks are intentionally NOT in NODES.
"""

import argparse, json, math, os, posixpath, re, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    import paramiko
except ImportError:
    sys.exit("Missing dependency: run  `py -m pip install paramiko`  first.")

# ============================ CONFIG  (edit these) ====================================
# host  = management IP the WINDOWS box reaches the server on (SSH/SFTP)
# fast  = IP on the SEPARATE 10Gbps inter-server subnet (used by run-mpi only)
# workers / numa_nodes : an integer, or "auto" (detected via `probe`)
# auth  : set EITHER "key" (path to a private key on Windows) OR "password".
NODES = [
    {"name": "numa", "host": "192.168.51.73", "fast": "10.0.0.1",
     "user": "root", "key": None, "password": None, "workers": "auto", "numa_nodes": "auto"},
    {"name": "aziz", "host": "192.168.7.138", "fast": "10.0.0.2",
     "user": "aziz", "key": None, "password": None, "workers": "auto", "numa_nodes": "auto"},
    {"name": "srv3", "host": "PUT.THIRD.SERVER.IP", "fast": "10.0.0.3",
     "user": "USER", "key": None, "password": None, "workers": "auto", "numa_nodes": "auto"},
]

REMOTE_DIR    = "pmmd_rigidity"          # created under each server's $HOME
LOCAL_WORKER  = "foam_rigidity.py"
RUN_TAG       = time.strftime("%Y%m%d-%H%M%S")

# experiment grid (finite-size scaling toward the continuum):
N_SCHEDULE    = [20000, 50000, 100000, 200000, 400000]
SAMPLES_PER_N = 64
WORKER_PARAMS = "--z 240 --pc 1.0 --foam e8 --dfix 4.0 --probes 48 --m 80 --eps 0.05 --kfreqs 0.5,1.0,1.5"

# --- auto-tuning knobs ---
RESERVE_FRAC  = 0.12      # keep >=12% of total RAM for the OS
RESERVE_GB    = 2.0       # ...but at least 2 GB
MEM_SAFETY    = 1.6       # safety factor on the per-worker memory estimate
PWMB_BASE     = 160.0     # per-worker baseline MB (python+numpy+scipy)
PWMB_PER_MN   = 22000.0   # extra MB per 1e6 points at z=240 (from calibration ~0.022 MB/pt)
# =====================================================================================

PHI = (1 + 5 ** 0.5) / 2
TARGET_G_FOAM = 1.0 / (4 * PHI ** 10)
TARGET_RATIO = 2 * PHI ** 5


def _z_param():
    m = re.search(r"--z\s+(\d+)", WORKER_PARAMS)
    return int(m.group(1)) if m else 240


def per_worker_mb(maxN, z):
    return MEM_SAFETY * (PWMB_BASE + PWMB_PER_MN * (maxN / 1e6) * (z / 240.0))


# ------------------------------- ssh plumbing ----------------------------------------
def connect(node, timeout=20):
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    kw = dict(hostname=node["host"], username=node["user"], timeout=timeout,
              banner_timeout=timeout, auth_timeout=timeout)
    if node.get("key"):
        kw["key_filename"] = os.path.expanduser(node["key"])
    elif node.get("password"):
        kw["password"] = node["password"]
    c.connect(**kw)
    return c


def run(c, cmd, get_pty=False):
    si, so, se = c.exec_command(cmd, get_pty=get_pty)
    out = so.read().decode("utf-8", "replace")
    err = se.read().decode("utf-8", "replace")
    return so.channel.recv_exit_status(), out, err


def rdir(c):
    _, out, _ = run(c, "echo $HOME")
    return posixpath.join(out.strip(), REMOTE_DIR)


def sftp_put(c, local, remote):
    sf = c.open_sftp()
    run(c, "mkdir -p '%s'" % posixpath.dirname(remote))
    sf.put(local, remote); sf.close()


def parallel(fn, nodes=NODES):
    res = {}
    with ThreadPoolExecutor(max_workers=len(nodes)) as ex:
        futs = {ex.submit(fn, n): n for n in nodes}
        for f in as_completed(futs):
            n = futs[f]
            try:
                res[n["name"]] = f.result()
            except Exception as e:
                res[n["name"]] = "ERROR: %s" % e
    for n in nodes:
        print("  [%-6s] %s" % (n["name"], res.get(n["name"], "")))
    return res


# ------------------------------- detection / auto-tuning -----------------------------
_DETECT = (
    "echo @CPU; LC_ALL=C lscpu 2>/dev/null | awk -F: '"
    "/^Socket\\(s\\)/{s=$2}/^NUMA node\\(s\\)/{n=$2}/^Core\\(s\\) per socket/{c=$2}"
    "END{gsub(/ /,\"\",s);gsub(/ /,\"\",n);gsub(/ /,\"\",c);print s,n,c}'; "
    "echo @MEM; awk '/MemTotal/{t=$2}/MemAvailable/{a=$2}END{print t,a}' /proc/meminfo; "
    "echo @NPROC; nproc"
)


def detect_node(n):
    c = connect(n)
    try:
        _, out, _ = run(c, _DETECT)
    finally:
        c.close()
    sockets = numa = cps = nproc = None
    lines = [l.strip() for l in out.splitlines()]
    for i, l in enumerate(lines):
        if l == "@CPU" and i + 1 < len(lines):
            parts = lines[i + 1].split()
            if len(parts) == 3 and all(p.isdigit() for p in parts):
                sockets, numa, cps = map(int, parts)
        elif l == "@MEM" and i + 1 < len(lines):
            t, a = lines[i + 1].split()
            mt, ma = int(t), int(a)
        elif l == "@NPROC" and i + 1 < len(lines):
            nproc = int(lines[i + 1])
    phys = (sockets * cps) if (sockets and cps) else (nproc or 1)
    return {"sockets": sockets or 1, "numa": numa or 1, "phys_cores": phys,
            "mem_total_kb": mt, "mem_avail_kb": ma}


def cmd_probe(args):
    print("== probe: detecting sockets / physical cores / free RAM ==")
    det = {}

    def do(n):
        d = detect_node(n)
        det[n["name"]] = d
        return ("sockets=%(sockets)d numa=%(numa)d phys_cores=%(phys_cores)d "
                "RAM=%(mem_total_kb).0f kB avail=%(mem_avail_kb).0f kB" % d)

    parallel(do)
    json.dump(det, open("nodes_detected.json", "w"), indent=2)
    z = _z_param(); pw = per_worker_mb(max(N_SCHEDULE), z)
    print("per-worker memory budget @ maxN=%d, z=%d : ~%.0f MB" % (max(N_SCHEDULE), z, pw))
    for n in NODES:
        if n["name"] in det:
            print("  [%-6s] auto workers -> %d, numa -> %d"
                  % (n["name"], _effective(n, det)[0], _effective(n, det)[1]))
    print("cached -> nodes_detected.json")


def _load_detected():
    if os.path.exists("nodes_detected.json"):
        return json.load(open("nodes_detected.json"))
    return {}


def _ram_cap_workers(nd, maxN, z):
    mt, ma = nd.get("mem_total_kb"), nd.get("mem_avail_kb")
    if not mt or not ma:
        return nd.get("phys_cores", 1)
    reserve = max(mt * RESERVE_FRAC, RESERVE_GB * 1048576)
    usable = max(ma - reserve, 0)
    pw_kb = per_worker_mb(maxN, z) * 1024
    return max(int(usable / pw_kb), 1)


def _effective(n, det):
    nd = det.get(n["name"], {})
    phys = nd.get("phys_cores") or 1
    numa = nd.get("numa") or 1
    w, nn = n["workers"], n["numa_nodes"]
    z, maxN = _z_param(), max(N_SCHEDULE)
    if w == "auto":
        w = min(phys, _ram_cap_workers(nd, maxN, z)) if nd else phys
    if nn == "auto":
        nn = numa
    return int(w), int(nn)


def _ensure_detected():
    det = _load_detected()
    need = any(n["workers"] == "auto" or n["numa_nodes"] == "auto" for n in NODES)
    if need and not det:
        print("(auto config but no nodes_detected.json -> probing first)")
        det = {}
        for n in NODES:
            try:
                det[n["name"]] = detect_node(n)
            except Exception as e:
                print("  probe %s failed: %s" % (n["name"], e))
        json.dump(det, open("nodes_detected.json", "w"), indent=2)
    return det


# ------------------------------- setup / gen -----------------------------------------
def cmd_setup(args):
    print("== setup: deploy worker + venv + deps ==")

    def do(n):
        c = connect(n)
        try:
            rd = rdir(c)
            run(c, "mkdir -p '%s/results'" % rd)
            sftp_put(c, LOCAL_WORKER, posixpath.join(rd, LOCAL_WORKER))
            steps = ("cd '%s' && (python3 -m venv venv || true) && "
                     "./venv/bin/python -m pip -q install --upgrade pip && "
                     "./venv/bin/python -m pip -q install numpy scipy %s && "
                     "./venv/bin/python -c 'import numpy,scipy;print(numpy.__version__,scipy.__version__)'"
                     % (rd, "mpi4py" if args.mpi else ""))
            rc, out, err = run(c, steps)
            return ("ok numpy/scipy " + out.strip()) if rc == 0 else "FAILED: " + (err or out).strip()[:200]
        finally:
            c.close()

    parallel(do)


def _partition(units, weights):
    tot = sum(weights.values())
    out, i, names = {}, 0, list(weights)
    for k, name in enumerate(names):
        share = len(units) - i if k == len(names) - 1 else round(len(units) * weights[name] / tot)
        out[name] = units[i:i + share]; i += share
    return out


def cmd_gen(args):
    det = _ensure_detected()
    weights = {n["name"]: _effective(n, det)[0] for n in NODES}
    units = [(N, s) for N in N_SCHEDULE for s in range(SAMPLES_PER_N)]
    parts = _partition(units, weights)
    d = "run_%s" % RUN_TAG; os.makedirs(d, exist_ok=True)
    for name, u in parts.items():
        open(os.path.join(d, "queue_%s.txt" % name), "w").write(
            "\n".join("%d,%d" % (N, s) for N, s in u) + "\n")
    json.dump({"tag": RUN_TAG, "weights": weights, "total": len(units),
               "partition": {k: len(v) for k, v in parts.items()}},
              open("manifest.json", "w"), indent=2)
    print("== gen: %d units (auto-tuned weights=%s) ==" % (len(units), weights))
    for name, u in parts.items():
        print("  [%-6s] %d units" % (name, len(u)))


# RAM-aware, NUMA-bound launcher deployed to each server
_RUN_NODE_SH = r"""#!/usr/bin/env bash
set -u
DIR="$1"; PY="$2"; REQW="$3"; NUMA="$4"; PWMB="$5"; RFRAC="$6"; RGB="$7"; shift 7; PARAMS="$@"
cd "$DIR"; mkdir -p results; touch queue.lock
MT=$(awk '/MemTotal/{print $2}' /proc/meminfo)
MA=$(awk '/MemAvailable/{print $2}' /proc/meminfo)
RES=$(awk -v t="$MT" -v f="$RFRAC" -v g="$RGB" 'BEGIN{r=t*f; gg=g*1048576; if(gg>r)r=gg; printf "%d",r}')
USABLE=$(( MA - RES )); [ "$USABLE" -lt 0 ] && USABLE=0
PWKB=$(awk -v m="$PWMB" 'BEGIN{printf "%d", m*1024}'); [ "$PWKB" -lt 1 ] && PWKB=1
MAXR=$(( USABLE / PWKB ))
WORKERS=$REQW; [ "$MAXR" -lt "$WORKERS" ] && WORKERS=$MAXR; [ "$WORKERS" -lt 1 ] && WORKERS=1
if [ "$NUMA" -le 0 ]; then NUMA=$(LC_ALL=C lscpu | awk -F: '/^NUMA node\(s\)/{gsub(/ /,"",$2);print $2}'); fi
[ -z "$NUMA" ] && NUMA=1; [ "$NUMA" -lt 1 ] && NUMA=1
echo "[autotune] MemAvail=${MA}kB reserve=${RES}kB per_worker=${PWMB}MB -> workers=$WORKERS (req $REQW) numa=$NUMA"
consume() {
  local wid="$1"
  while true; do
    line=$(flock queue.lock bash -c 'head -n1 queue.txt 2>/dev/null; sed -i "1d" queue.txt 2>/dev/null')
    [ -z "$line" ] && break
    N=${line%%,*}; S=${line##*,}; OUT="results/${N}_${S}.json"
    [ -s "$OUT" ] && continue
    BIND=""; if [ "$NUMA" -gt 1 ]; then s=$((wid % NUMA)); BIND="numactl --cpunodebind=$s --membind=$s"; fi
    $BIND "$PY" foam_rigidity.py --N "$N" --seed "$S" $PARAMS --out "$OUT" >>"worker_${wid}.log" 2>&1
  done
}
for w in $(seq 0 $((WORKERS-1))); do consume "$w" & done
wait; echo ALL_DONE > .done
"""


def cmd_run(args):
    det = _ensure_detected()
    qdir = "run_%s" % (args.tag or _latest("run_"))
    if not os.path.isdir(qdir):
        sys.exit("No work dir %s ; run `gen` first." % qdir)
    z, maxN = _z_param(), max(N_SCHEDULE)
    pwmb = per_worker_mb(maxN, z)
    print("== run: detached RAM-aware pools (per-worker budget ~%.0f MB) ==" % pwmb)

    def do(n):
        c = connect(n)
        try:
            rd = rdir(c); pyx = posixpath.join(rd, "venv/bin/python")
            ql = os.path.join(qdir, "queue_%s.txt" % n["name"])
            if not os.path.exists(ql):
                return "no queue (skipped)"
            sftp_put(c, ql, posixpath.join(rd, "queue.txt"))
            sf = c.open_sftp()
            with sf.open(posixpath.join(rd, "run_node.sh"), "w") as fh:
                fh.write(_RUN_NODE_SH)
            sf.close(); run(c, "chmod +x '%s/run_node.sh'" % rd)
            reqw, numa = _effective(n, det)
            cmd = ("cd '%s' && rm -f .done && setsid nohup bash run_node.sh '%s' '%s' %d %d %.1f %.3f %.1f %s "
                   ">run.log 2>&1 < /dev/null & echo $! > run.pid; cat run.pid"
                   % (rd, rd, pyx, reqw, numa, pwmb, RESERVE_FRAC, RESERVE_GB, WORKER_PARAMS))
            _, out, _ = run(c, cmd)
            return "started pid=%s, requested workers=%d, numa=%d (live RAM cap on node)" % (out.strip(), reqw, numa)
        finally:
            c.close()

    parallel(do)
    print("running. Check `py farm.py status` (each node prints its [autotune] line in run.log).")


def cmd_status(args):
    print("== status ==  target G_foam=%.4e (ell_*/ell_P=%.3f)" % (TARGET_G_FOAM, TARGET_RATIO))

    def do(n):
        c = connect(n)
        try:
            rd = rdir(c)
            _, done, _ = run(c, "ls '%s'/results/*.json 2>/dev/null | wc -l" % rd)
            _, left, _ = run(c, "wc -l < '%s'/queue.txt 2>/dev/null || echo 0" % rd)
            _, proc, _ = run(c, "pgrep -fc foam_rigidity.py || echo 0")
            _, tune, _ = run(c, "grep -h autotune '%s'/run.log 2>/dev/null | tail -1" % rd)
            return "done=%-6s left=%-6s running=%-3s | %s" % (
                done.strip(), left.strip(), proc.strip(), tune.strip())
        finally:
            c.close()

    parallel(do)


def cmd_collect(args):
    out = os.path.join("collected", args.tag or RUN_TAG); os.makedirs(out, exist_ok=True)
    print("== collect -> %s ==" % out)

    def do(n):
        c = connect(n)
        try:
            rd = rdir(c); sf = c.open_sftp(); rp = posixpath.join(rd, "results")
            try:
                files = sf.listdir(rp)
            except IOError:
                return "no results"
            cnt = 0
            for f in files:
                if f.endswith(".json"):
                    sf.get(posixpath.join(rp, f), os.path.join(out, "%s__%s" % (n["name"], f))); cnt += 1
            sf.close(); return "pulled %d json" % cnt
        finally:
            c.close()

    parallel(do)


def cmd_aggregate(args):
    import numpy as np
    folder = os.path.join("collected", args.tag) if args.tag else os.path.join("collected", _latest("", base="collected"))
    files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".json")]
    by_N = {}
    for fp in files:
        try:
            d = json.load(open(fp))
        except Exception:
            continue
        if d.get("status") == "ok":
            by_N.setdefault(d["N_target"], []).append(d)
    if not by_N:
        sys.exit("No OK results in %s" % folder)
    print("== aggregate (%s) ==" % folder)
    print("%-9s %7s %13s %12s %10s" % ("N", "samples", "G_normed mean", "sem", "spec_dim"))
    rows = []
    for N in sorted(by_N):
        ds = by_N[N]
        g = np.array([x["G_foam_normed"] for x in ds], float)
        sd = np.array([x.get("spectral_dim_measured", float("nan")) for x in ds], float)
        gm = g.mean(); gs = g.std(ddof=1) / math.sqrt(len(g)) if len(g) > 1 else 0.0
        rows.append((N, gm, gs, np.nanmean(sd)))
        print("%-9d %7d %13.5g %12.3g %10.4f" % (N, len(ds), gm, gs, np.nanmean(sd)))
    Ns = np.array([r[0] for r in rows], float); Gs = np.array([r[1] for r in rows], float)
    if len(Ns) >= 2:
        A = np.vstack([np.ones_like(Ns), Ns ** -0.5]).T
        G_inf = np.linalg.lstsq(A, Gs, rcond=None)[0][0]
        print("\nFSS  G_foam_normed(N->inf) = %.6g    target 1/(4 phi^10) = %.6g" % (G_inf, TARGET_G_FOAM))
        print("calibration constant to match: Gnorm_fit = %.4g  (rerun workers with --Gnorm this value)"
              % (TARGET_G_FOAM / G_inf if G_inf else float("nan")))
    print("\nNOTE: matching the ABSOLUTE 1/(4 phi^10) fixes the scheme constant Gnorm [H3];")
    print("the spectral dimension (should agree between --foam poisson and --foam e8) and the")
    print("FSS trend are the physical observables already produced by the farm.")


def cmd_runmpi(args):
    total = sum(_effective(n, _ensure_detected())[0] for n in NODES)
    head = NODES[0]; c = connect(head)
    try:
        rd = rdir(c); pyx = posixpath.join(rd, "venv/bin/python")
        hf = "\n".join("%s slots=%d" % (n["fast"], _effective(n, _load_detected())[0]) for n in NODES) + "\n"
        sf = c.open_sftp()
        with sf.open(posixpath.join(rd, "hostfile"), "w") as fh:
            fh.write(hf)
        sf.close()
        cmd = ("cd '%s' && for S in $(seq 0 %d); do mpirun --hostfile hostfile -np %d "
               "--mca btl_tcp_if_include %s %s foam_rigidity.py --N %d --seed $S --mpi %s "
               "--out results/mpi_%d_${S}.json; done > run_mpi.log 2>&1 &"
               % (rd, args.samples - 1, total, args.fast_iface, pyx, args.N, WORKER_PARAMS, args.N))
        run(c, cmd)
        print("== run-mpi launched on %s over fast subnet: np=%d, N=%d, samples=%d ==" %
              (head["name"], total, args.N, args.samples))
    finally:
        c.close()


def cmd_stop(args):
    print("== stop ==")

    def do(n):
        c = connect(n)
        try:
            rd = rdir(c)
            run(c, "cd '%s' 2>/dev/null && [ -f run.pid ] && kill -- -$(cat run.pid) 2>/dev/null || true" % rd)
            run(c, "pkill -f run_node.sh 2>/dev/null; pkill -f foam_rigidity.py 2>/dev/null; true")
            return "stopped"
        finally:
            c.close()

    parallel(do)


def _latest(prefix, base="."):
    items = sorted(d[len(prefix):] for d in os.listdir(base)
                   if d.startswith(prefix) and os.path.isdir(os.path.join(base, d)))
    if not items:
        sys.exit("nothing matching %r in %s" % (prefix, base))
    return items[-1]


def main():
    ap = argparse.ArgumentParser(description="PMMD foam-rigidity farm console (run on Windows).")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("probe")
    s = sub.add_parser("setup"); s.add_argument("--mpi", action="store_true")
    sub.add_parser("gen")
    s = sub.add_parser("run"); s.add_argument("--tag", default=None)
    sub.add_parser("status")
    s = sub.add_parser("collect"); s.add_argument("--tag", default=None)
    s = sub.add_parser("aggregate"); s.add_argument("--tag", default=None)
    s = sub.add_parser("run-mpi"); s.add_argument("--N", type=int, required=True)
    s.add_argument("--samples", type=int, default=8); s.add_argument("--fast-iface", default="10.0.0.0/24")
    sub.add_parser("stop")
    args = ap.parse_args()
    {"probe": cmd_probe, "setup": cmd_setup, "gen": cmd_gen, "run": cmd_run, "status": cmd_status,
     "collect": cmd_collect, "aggregate": cmd_aggregate, "run-mpi": cmd_runmpi, "stop": cmd_stop}[args.cmd](args)


if __name__ == "__main__":
    main()
