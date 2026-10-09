#!/bin/bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause-Clear
#
# The per-cap tables for this machine: gtc_table[GTC_CAP (I)] for I = 2..5,
# then r2gtc_table[GTC_CAP (I)] for I = 2..5 over them, printed as
# gtc_config takes them (and as src/config.h holds the defaults).  Each cap
# is tuned twice (tune_gauss --table / --rat-table --cap I), with the lower
# caps' tables as found before; its table is the sizes where both passes
# chose the same algorithm and that is not the thresholds' choice.  Takes
# a few hours.  Run it on an otherwise idle machine:
#
#   tools/tune.sh [CPU]          (CPU to pin to, default 0)
#
# B (build directory, default build) and OUT (the tuner's output,
# default tune-out) can be set in the environment; make's variables too,
# e.g. GMP_PREFIX.

set -eu
CPU=${1:-0}
B=${B:-build}
OUT=${OUT:-tune-out}
mkdir -p "$OUT"
echo "=== $(uname -m), cpu $CPU"; date
make -j2 B="$B" "$B/tune_gauss" >/dev/null

# The sizes marked "(table)" with the same choice in both files, as a
# table list.
merge () {
  python3 - "$1" "$2" <<'EOF'
import re, sys
def read(f):
    ch = {}
    for l in open(f):
        m = re.match(r'\s*(\d+)\s.*\s(\S) \(table\)$', l)
        if m:
            ch[int(m[1])] = m[2]
    return ch
a, b = read(sys.argv[1]), read(sys.argv[2])
keep = {n: c for n, c in a.items() if b.get(n) == c}
ns = sorted(keep); out = []; i = 0
while i < len(ns):
    j = i
    while (j + 1 < len(ns) and ns[j + 1] == ns[j] + 1
           and keep[ns[j + 1]] == keep[ns[i]]):
        j += 1
    out.append(f"{ns[i]}-{ns[j]}:{keep[ns[i]]}" if j > i
               else f"{ns[i]}:{keep[ns[i]]}")
    i = j + 1
print(' '.join(out))
EOF
}

declare -A G R
for I in 2 3 4 5; do
  SET=()
  for J in 2 3 4 5; do
    if [ $J -lt $I ]; then SET+=(--set-table $J "${G[$J]}"); fi
  done
  # Above the textbook limit a cap-2 product has no choice.
  N=1024; [ $I = 2 ] && N=24
  for p in 1 2; do
    "./$B/tune_gauss" --pin "$CPU" --table $N --cap $I "${SET[@]}" \
      > "$OUT/g$I.$p"
  done
  G[$I]=$(merge "$OUT/g$I.1" "$OUT/g$I.2")
  echo "gtc_table[GTC_CAP ($I)] \"${G[$I]}\""; date
done
GSET=()
for J in 2 3 4 5; do GSET+=(--set-table $J "${G[$J]}"); done
for I in 2 3 4 5; do
  SET=()
  for J in 2 3 4 5; do
    if [ $J -lt $I ]; then SET+=(--set-rat-table $J "${R[$J]}"); fi
  done
  for p in 1 2; do
    "./$B/tune_gauss" --pin "$CPU" --rat-table 2:1023 --cap $I \
      "${GSET[@]}" "${SET[@]}" > "$OUT/r$I.$p"
  done
  R[$I]=$(merge "$OUT/r$I.1" "$OUT/r$I.2")
  echo "r2gtc_table[GTC_CAP ($I)] \"${R[$I]}\""; date
done
echo "done"; date
