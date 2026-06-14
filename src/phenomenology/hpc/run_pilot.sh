#!/usr/bin/env bash
# 2D pilot for ONE coupling value g (default 1.0). Independent run -> distribute g across machines.
#   usage:  bash run_pilot.sh <g>
# Output: cvals_g<g>.dat (the a,|c| points) and result_g<g>.txt (continuum |c|_0 for this g).
set -e
G=${1:-1.0}; B=3; BOX=8.0
TAG="g${G}"
: > cvals_${TAG}.dat
for L in 96 128 160 192; do
  echo "### g=$G L=$L ###"
  python3 stage_A_soliton.py --L $L --B $B --mu 0.3 --kappa 1.0 --steps 30000 --box $BOX --out sol_${TAG}_L${L}.npy
  # check the printed Berg-Luscher Q == -3 and E4/E0 ~ 1 before trusting this L
  C=$(python3 stage_BC_overlap.py --soliton sol_${TAG}_L${L}.npy --g $G --nmodes 8 --B $B \
        | awk -F'= ' '/\|c\| =/{print $2}' | awk '{print $1}')
  A=$(python3 -c "print($BOX*2/$L)")
  echo "$A $C" >> cvals_${TAG}.dat
done
echo "g=$G continuum:" | tee result_${TAG}.txt
python3 stage_D_continuum.py cvals_${TAG}.dat | tee -a result_${TAG}.txt
