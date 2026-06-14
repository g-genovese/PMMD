#!/usr/bin/env bash
# =============================================================================
# PMMD pilot orchestrator (venv + NUMA + Linux/Windows-domain aware).
# Run from ONE Linux machine. Deploys the pilot, creates/uses a Python venv on
# each node, launches one (or more, NUMA-pinned) coupling g per node IN PARALLEL
# over SSH, then collects results.
#
# PREREQUISITES (one-time): passwordless SSH (keys) from THIS machine to every node.
#   Linux node: python3 + venv module present (apt install python3-venv if needed);
#               numactl present if you use NUMA bindings.
#   Windows node (domain): OpenSSH Server enabled, default shell = cmd, python on PATH.
#               USER must be the DOMAIN user written as  DOMAIN\\user  (double backslash)
#               or UPN  user@domain.local .
# =============================================================================

# ----------------------------- CONFIG: edit here -----------------------------
VENV="rs"                              # venv base folder: $HOME/rs (Linux) / %USERPROFILE%\rs (Win)
REMOTE_DIR="pmmd_hpc_pilot"            # code folder, relative to remote home (both OS)
DEPS="jax scipy numpy"                 # modules needed by the calculations (CPU jax is fine for 2D)
LOCAL_SRC="$(cd "$(dirname "$0")" && pwd)"
RESULTS_DIR="./scan_results"
SSH_OPTS="-o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10 -o ServerAliveInterval=15 -o ServerAliveCountMax=4"

# fields:  NAME | SSH_IP | USER | OS(linux|windows) | TASKS | INTERNAL_10G_IP
# TASKS = comma-separated list. Each task is  g   or   g@numanode  (numanode -> separate
#         ssh with numactl --cpunodebind=N --membind=N). Server C is 2-socket -> 2 tasks.
NODES=(
  "serverA|192.168.1.11|gianluca|linux|0.5|10.10.0.11"
  "serverB|192.168.1.12|gianluca|linux|0.75|10.10.0.12"
  "serverC|192.168.1.13|gianluca|linux|1.5@0,1.6@1|10.10.0.13"
  "workstation|192.168.1.20|DOMAIN\\\\gianluca|windows|2.0|-"
  "nb1|192.168.1.31|gianluca|linux|1.25|-"
  "nb2|192.168.1.32|gianluca|linux|1.0|-"
)
# -----------------------------------------------------------------------------

mkdir -p "$RESULTS_DIR"
field(){ echo "$1" | cut -d'|' -f"$2"; }

# remote command fragments -----------------------------------------------------
venv_create_linux="test -d \$HOME/$VENV || python3 -m venv \$HOME/$VENV; . \$HOME/$VENV/bin/activate; python -m pip install --upgrade pip $DEPS"
venv_create_win="if not exist %USERPROFILE%\\$VENV ( python -m venv %USERPROFILE%\\$VENV ) & call %USERPROFILE%\\$VENV\\Scripts\\activate.bat & python -m pip install --upgrade pip $DEPS"

run_linux(){  # $1=numa(empty|N) $2=g
  local numa="" ; [ -n "$1" ] && numa="numactl --cpunodebind=$1 --membind=$1 "
  printf '%s' ". \$HOME/$VENV/bin/activate && cd \$HOME/$REMOTE_DIR && ${numa}python run_pilot.py $2"
}
run_win(){    # $1=g  (numactl n/a on Windows)
  printf '%s' "call %USERPROFILE%\\$VENV\\Scripts\\activate.bat & cd /d %USERPROFILE%\\$REMOTE_DIR & python run_pilot.py $1"
}

deploy(){
  for e in "${NODES[@]}"; do
    n=$(field "$e" 1); ip=$(field "$e" 2); u=$(field "$e" 3); os=$(field "$e" 4)
    echo "[deploy] $n ($ip,$os)"
    echo "  - probe ssh ..."
    if ! ssh $SSH_OPTS "$u@$ip" "echo ok" >/dev/null 2>&1; then
      echo "  !! cannot ssh to $u@$ip (check IP/keys/firewall). SKIPPING."; continue
    fi
    if [ "$os" = windows ]; then
      echo "  - copying code ..."; scp $SSH_OPTS -r "$LOCAL_SRC" "$u@$ip:$REMOTE_DIR" >/dev/null
      [ "$1" = --setup ] && { echo "  - creating venv + installing deps (minutes; verbose) ..."; ssh $SSH_OPTS "$u@$ip" "$venv_create_win"; }
    else
      echo "  - mkdir + copying code ..."; ssh $SSH_OPTS "$u@$ip" "mkdir -p \$HOME/$REMOTE_DIR"
      scp $SSH_OPTS -r "$LOCAL_SRC"/* "$u@$ip:$REMOTE_DIR/" >/dev/null
      [ "$1" = --setup ] && { echo "  - creating venv + installing deps (minutes; verbose) ..."; ssh $SSH_OPTS "$u@$ip" "bash -lc '$venv_create_linux'"; }
    fi
    echo "  - $n ready."
  done
}

launch(){
  echo "[launch] one ssh per task (NUMA tasks are separate ssh, as required)..."
  PIDS=()
  for e in "${NODES[@]}"; do
    n=$(field "$e" 1); ip=$(field "$e" 2); u=$(field "$e" 3); os=$(field "$e" 4); tasks=$(field "$e" 5)
    IFS=',' read -ra TS <<< "$tasks"
    for t in "${TS[@]}"; do
      g="${t%@*}"; numa=""; [[ "$t" == *@* ]] && numa="${t#*@}"
      log="$RESULTS_DIR/${n}_g${g}${numa:+_numa$numa}.log"
      if [ "$os" = windows ]; then
        ssh $SSH_OPTS "$u@$ip" "$(run_win "$g")" > "$log" 2>&1 &
      else
        ssh $SSH_OPTS "$u@$ip" "bash -lc '$(run_linux "$numa" "$g")'" > "$log" 2>&1 &
      fi
      PIDS+=($!); echo "  $n -> g=$g ${numa:+(numa $numa)} pid $!"
    done
  done
  echo "[launch] waiting..."; wait "${PIDS[@]}"; echo "[launch] done."
}

collect(){
  echo "[collect] fetching results..."
  for e in "${NODES[@]}"; do
    n=$(field "$e" 1); ip=$(field "$e" 2); u=$(field "$e" 3); os=$(field "$e" 4); tasks=$(field "$e" 5)
    IFS=',' read -ra TS <<< "$tasks"
    for t in "${TS[@]}"; do
      g="${t%@*}"
      scp $SSH_OPTS "$u@$ip:$REMOTE_DIR/result_g${g}.txt" "$RESULTS_DIR/${n}_result_g${g}.txt" 2>/dev/null \
        && echo "  got $n g=$g" || echo "  MISSING $n g=$g (see $RESULTS_DIR/${n}_g${g}*.log)"
    done
  done
  echo; echo "============ SCAN SUMMARY (g -> continuum |c|_0) ============"
  cat "$RESULTS_DIR"/*_result_g*.txt 2>/dev/null
  echo "============================================================="
}

case "${1:-all}" in
  deploy)  deploy "$2";;
  launch)  launch;;
  collect) collect;;
  all)     deploy "$2"; launch; collect;;
  *) echo "usage: $0 [all|deploy|launch|collect] [--setup]";;
esac
