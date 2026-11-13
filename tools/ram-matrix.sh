#!/bin/bash
# TinyTitan RAM matrix: decode tok/s for both Qwen3.8 quant models across --ram budgets.
# Usage (from the repo root):  bash tools/ram-matrix.sh
# Env overrides: BUDGETS="8 10 12 14 16 18 default"  MAX_NEW=200  RAM_FLAG=--ram-budget
#                MODELS="qwen3.8-flash-next_125B_A6B_4Bit qwen3.8-flash-next_125B_A6B_4Bit_BF16Rest"
# Writes: .build/ram-matrix/RESULTS-<timestamp>.md (table updated after every run) + per-run .out/.err
# Prints the same table rows to the terminal as each run finishes.

cd "$(dirname "$0")/.." 2>/dev/null || true
[ -x .build/release/TinyTitanCLI ] || { echo "error: .build/release/TinyTitanCLI not found; run from the TinyTitan repo or build release first"; exit 1; }

BUDGETS=${BUDGETS:-"8 10 12 14 16 18 default"}
MODELS=${MODELS:-"qwen3.8-flash-next_125B_A6B_4Bit"}
MAX_NEW=${MAX_NEW:-200}
RAM_FLAG=${RAM_FLAG:---ram-budget}
OUT=.build/ram-matrix
STAMP=$(date -u +%Y%m%d-%H%M%S)
MD=$OUT/RESULTS-$STAMP.md
mkdir -p "$OUT"

P1="Explain what a mutex is and when you would use one."
if [ -f .build/task12-swift-prompt.txt ]; then
  P2=$(cat .build/task12-swift-prompt.txt); P2NAME="swift-rename"
else
  P2="Write a Swift function that renames a git branch using the Gitea REST API, with error handling."; P2NAME="swift-gitea"
fi

# Wait for other benchmark jobs so runs don't compete for memory bandwidth.
while screen -ls 2>/dev/null | grep -q -E "tt16|ttgolden|ttmatrix-other"; do
  echo "waiting for running screen jobs (tt16/ttgolden) to finish..."; sleep 30
done

{
  echo "# TinyTitan RAM matrix — $(date '+%Y-%m-%d %H:%M')"
  echo
  echo "Commit \`$(git rev-parse --short HEAD 2>/dev/null)\` · $MAX_NEW tokens greedy (temp 0, seed 1234) · expert profile off · $(sysctl -n hw.model 2>/dev/null) $(( $(sysctl -n hw.memsize 2>/dev/null || echo 0) / 1073741824 )) GB"
  echo
  echo "| RAM | Model | Prompt | Slots | Hit rate | Decode tok/s | Wall s | Result |"
  echo "|---|---|---|---|---|---|---|---|"
} > "$MD"

printf "%-8s %-9s %-13s %-6s %-8s %-8s %-7s %s\n" RAM MODEL PROMPT SLOTS HIT TOK/S WALL RESULT
total=0; for b in $BUDGETS; do for m in $MODELS; do total=$((total+2)); done; done
n=0

for b in $BUDGETS; do
  for pi in 1 2; do
    if [ $pi = 1 ]; then P=$P1; PN=mutex; else P=$P2; PN=$P2NAME; fi
    for m in $MODELS; do
      n=$((n+1))
      short=${m##*_}                     # 4Bit / BF16Rest
      f=$OUT/$STAMP-$b-$short-$PN
      if [ "$b" = default ]; then RB=""; else RB="$RAM_FLAG ${b}G"; fi
      t0=$(date +%s)
      # shellcheck disable=SC2086
      TINYTITAN_RUNNER_STATS=1 TINYTITAN_EXPERT_PROFILE=0 \
        .build/release/TinyTitanCLI --model "models/$m" --prompt "$P" \
        --max-new "$MAX_NEW" --temperature 0 --seed 1234 $RB > "$f.out" 2> "$f.err"
      rc=$?
      wall=$(( $(date +%s) - t0 ))
      slots=$(grep -hoE 'slots=[0-9]+' "$f.err" "$f.out" | head -1 | cut -d= -f2)
      hit=$(grep -hoE 'expert_hit_rate=[0-9.]+' "$f.err" "$f.out" | tail -1 | cut -d= -f2)
      tps=$(grep -hoE 'tok/s=[0-9.]+' "$f.err" "$f.out" | tail -1 | cut -d= -f2)
      [ -z "$tps" ] && tps=$(grep -hoiE '[0-9]+\.[0-9]+ *tok/s' "$f.err" "$f.out" | tail -1 | grep -oE '[0-9]+\.[0-9]+')
      if [ $rc -eq 0 ] && [ -n "$tps" ]; then
        result=OK
      else
        err=$(grep -hiE 'error|fatal|do not fit' "$f.err" | head -1 | tr '|' '/' | cut -c1-90)
        result="FAIL (rc=$rc) ${err:-see $f.err}"; tps=${tps:-—}
      fi
      echo "| $b | $short | $PN | ${slots:-—} | ${hit:-—} | **$tps** | $wall | $result |" >> "$MD"
      printf "%-8s %-9s %-13s %-6s %-8s %-8s %-7s %s   [%d/%d]\n" "$b" "$short" "$PN" "${slots:-—}" "${hit:-—}" "$tps" "$wall" "$result" $n $total
    done
  done
done

echo >> "$MD"
echo "Finished $(date '+%H:%M'). Raw output: \`$OUT/$STAMP-*.out|.err\`" >> "$MD"
echo
echo "Done. Markdown table: $MD"
