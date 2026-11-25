#!/usr/bin/env bash
#SBATCH -J real-pass-nofb
#SBATCH -p gpu-a100-dev
#SBATCH -N 1
#SBATCH --ntasks-per-node=3
#SBATCH --cpus-per-task=16
#SBATCH -A CCR26069
#SBATCH -t 02:00:00
#SBATCH -o /work/11808/shumera/ls6/butterfly_shared_resources/datasets/viz_program_eval_600/logs/real-pass-nofb.%j.out
#SBATCH -e /work/11808/shumera/ls6/butterfly_shared_resources/datasets/viz_program_eval_600/logs/real-pass-nofb.%j.out
# Real-image pass of all candidate programs with no fallback (see prepare_real_pass.py). One chunk per GPU.
# Resumable: generate.py reloads <chunk>/run/data.p, and chunks with DONE are skipped. Resubmit if incomplete.
# Run first:  python butterfly/metamorphic_eval/prepare_real_pass.py 3
OUT=${OUT:-/work/11808/shumera/ls6/butterfly_shared_resources/datasets/viz_program_eval_600/gqa-simple-v5/real_pass_nofallback}
export OUT REPO_ROOT=/work/11808/shumera/ls6/projects/butterfly START_S=$SECONDS WORKER_LIMIT=${LIMIT_SECONDS:-6900}
[ -f "$OUT/chunk_0/input.p" ] || { echo "Run prepare_real_pass.py first" >&2; exit 1; }
cat > "$OUT/worker.sh" <<'SH'
#!/usr/bin/env bash
source ~/.bashrc
cd "$REPO_ROOT"; source "$REPO_ROOT/butterfly/environment/activate.sh"
k=$SLURM_PROCID; c=$OUT/chunk_$k
export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID
export CONFIG_NAMES=config_codellama,viz_program_eval
export RAY_TEMP_DIR=$HOME/ray_${SLURM_JOB_ID}_$k; mkdir -p "$RAY_TEMP_DIR"
[ -f "$c/DONE" ] && exit 0
mkdir -p "$c/run"; R=0; [ -f "$c/run/data.p" ] && R=1
left=$((WORKER_LIMIT - (SECONDS-START_S)))
echo "[w$k gpu$CUDA_VISIBLE_DEVICES] start, ${left}s left, reload=$R"
timeout "$left" python -u generate.py --config_path "$c/config.yaml" --data_path "$c/input.p" --reload $R \
    > "$c/worker.log" 2>&1 && touch "$c/DONE" && echo "[w$k] done" || echo "[w$k] incomplete (rc=$?); resubmit to resume"
rm -rf "$RAY_TEMP_DIR"
SH
srun --ntasks=3 --ntasks-per-node=3 --cpus-per-task=16 --exclusive bash "$OUT/worker.sh"
echo "chunks done: $(ls -d "$OUT"/chunk_*/DONE 2>/dev/null | wc -l)/3"
