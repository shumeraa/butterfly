#!/usr/bin/env bash
#SBATCH -p gpu-a100-dev
#SBATCH --ntasks-per-node=3
#SBATCH -A CCR26069
#SBATCH -t 02:00:00
# Submitted by run_pipeline.sh (which sets -N, -J, -o). One job, all GPUs of its nodes (3x A100/node):
#   1 judge   one vLLM shard per node (tensor-parallel 2)     -> judge_run/labels.json
#   2 prepare pick transforms from the labels, split into chunks (CPU)
#   3 eval    one worker per GPU (3/node) runs chunks: render transforms, run all programs, score
#   4 export  comparison.jsonl + summary.md (+ hybrid analysis)
# The dev QOS allows ONE running job per user (max 2 nodes), so parallelism is inside this job.
# Resumable: resubmit via run_pipeline.sh; each stage/chunk skips what has a DONE marker.
PROJECT=/work/11808/shumera/ls6/projects/butterfly
: "${ROOT:?}" "${VERSION:?}" "${NAME:?}"
V=$ROOT/$NAME
LIMIT=${LIMIT_SECONDS:-7200}; MARGIN=300; START=$SECONDS
NNODES=${SLURM_NNODES:-1}
N_CHUNKS=${N_CHUNKS:-$((NNODES*3*3))}          # 3 chunks per worker
export WORK=/work/11808/shumera/ls6 REPO_ROOT=$PROJECT
cd "$PROJECT" || exit 1
echo "=== $NAME: prompt $VERSION, $NNODES node(s), $N_CHUNKS eval chunks ==="

# ---------- 1. judge ----------
if [ ! -f "$V/judge_run/DONE" ]; then
    mkdir -p "$V/judge_run"
    python3 - "$V" "$NNODES" <<'PY' || exit 1
import json, sys
from pathlib import Path
v, n = Path(sys.argv[1]), int(sys.argv[2])
rows = [l for l in open(v / 'questions.jsonl') if l.strip()]
for k in range(n):
    d = v / 'judge_run' / f'shard_{k}'; d.mkdir(parents=True, exist_ok=True)
    (d / 'questions.jsonl').write_text(''.join(rows[k::n]))
PY
    cat > "$V/judge_run/shard.sh" <<'SH'
#!/usr/bin/env bash
k=$SLURM_PROCID; d=$V/judge_run/shard_$k
args=(--questions "$d/questions.jsonl" --run-dir "$d/run" --max-model-len 4096 --prompt-version "$VERSION")
[ -f "$d/run/run_metadata.json" ] && args+=(--resume)
[ -f "$d/run/labels.json" ] && exit 0
bash "$REPO_ROOT/butterfly/llm_judge/run_inference.sh" "${args[@]}"
SH
    export V VERSION
    srun --nodes="$NNODES" --ntasks="$NNODES" --ntasks-per-node=1 --exclusive bash "$V/judge_run/shard.sh"
    python3 - "$V" "$NNODES" <<'PY' || { echo 'judge shards incomplete; resubmit to resume' >&2; exit 1; }
import json, sys
from pathlib import Path
v, n = Path(sys.argv[1]), int(sys.argv[2])
order = [json.loads(l)['id'] for l in open(v / 'questions.jsonl') if l.strip()]
labels = {}
for k in range(n):
    for qid, rec in json.load(open(v / 'judge_run' / f'shard_{k}' / 'run' / 'labels.json')):
        labels[qid] = rec
out = [[q, labels[q]] for q in order if q in labels]
json.dump(out, open(v / 'judge_run' / 'labels.json', 'w'))
with open(v / 'judge_run' / 'results.jsonl', 'w') as f:
    for k in range(n):
        f.write(open(v / 'judge_run' / f'shard_{k}' / 'run' / 'results.jsonl').read())
print(f'labels: {len(out)}/{len(order)} questions (the rest had no valid prediction)')
PY
    touch "$V/judge_run/DONE"
fi

# ---------- 2. prepare ----------
if [ ! -f "$V/metamorphic_eval/eval_ids.txt" ]; then
    source ~/.bashrc; source "$PROJECT/butterfly/environment/activate.sh"
    ONLY=""; [ "${TEST:-0}" = 1 ] && ONLY="$V/test_ids.txt"
    MERGED="$V" ONLY_IDS="$ONLY" python butterfly/metamorphic_eval/prepare_eval.py "$N_CHUNKS" 5 || exit 1
fi

# ---------- 3. eval: one worker per GPU ----------
cat > "$V/eval_worker.sh" <<'SH'
#!/usr/bin/env bash
# rank = global GPU slot; handles chunks k with k % WORKERS == rank
source ~/.bashrc
cd "$REPO_ROOT"; source "$REPO_ROOT/butterfly/environment/activate.sh"
rank=$SLURM_PROCID
export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID
export CONFIG_NAMES=config_codellama,viz_program_eval
export RAY_TEMP_DIR=$HOME/ray_${SLURM_JOB_ID}_$rank; mkdir -p "$RAY_TEMP_DIR"
slowest=0
for c in $(ls -d "$EVAL_DIR"/chunk_* | sort -t_ -k2 -n); do
    k=${c##*_}; [ $((k % WORKERS)) -eq "$rank" ] || continue
    [ -f "$c/DONE" ] && continue
    left=$((WORKER_LIMIT - (SECONDS-START_S)))
    need=$(( slowest > 0 ? slowest*11/10 : 900 ))
    [ "$left" -lt "$need" ] && { echo "[w$rank] stop: ${left}s left, need ~${need}s for chunk $k"; break; }
    mkdir -p "$c/run"; R=0; [ -f "$c/run/data.p" ] && R=1; t0=$SECONDS
    echo "[w$rank gpu$CUDA_VISIBLE_DEVICES] chunk $k start, ${left}s left"
    timeout "$left" python -u generate.py --config_path "$c/config.yaml" --data_path "$c/input.p" --reload $R \
        > "$c/worker.log" 2>&1 || { echo "[w$rank] chunk $k incomplete (rc=$?)"; break; }
    touch "$c/DONE"; dur=$((SECONDS-t0)); [ "$dur" -gt "$slowest" ] && slowest=$dur
    echo "[w$rank] chunk $k done in ${dur}s"
done
rm -rf "$RAY_TEMP_DIR"
SH
export EVAL_DIR=$V/metamorphic_eval
export WORKERS=$((NNODES*3))
export WORKER_LIMIT=$((LIMIT - (SECONDS-START) - MARGIN)) START_S=$SECONDS
srun --nodes="$NNODES" --ntasks="$WORKERS" --ntasks-per-node=3 --cpus-per-task=16 --exclusive bash "$V/eval_worker.sh"

# ---------- 4. export ----------
total=$(ls -d "$EVAL_DIR"/chunk_* | wc -l); done_n=$(ls -d "$EVAL_DIR"/chunk_*/DONE 2>/dev/null | wc -l)
echo "chunks done: $done_n/$total"
if [ "$done_n" -eq "$total" ]; then
    source ~/.bashrc; source "$PROJECT/butterfly/environment/activate.sh"
    python butterfly/metamorphic_eval/export_metamorphic_eval.py "$V" "$EVAL_DIR" \
        && python butterfly/metamorphic_eval/hybrid_analysis.py "$V" \
        && touch "$V/DONE" && echo "=== $NAME complete: $EVAL_DIR/summary.md ==="
else
    echo "Not finished: resubmit with run_pipeline.sh $VERSION to resume."
fi
