#!/bin/bash

# --- BATCH CONFIGURATION ---
# Sizable via env so you can make a small dataset to iterate quickly, then a big one:
#   CALOMAPS_NJOBS=40 CALOMAPS_GUN_PARTICLE=pi+ bash generate_batched.sh   # 40 files
TOTAL_JOBS="${CALOMAPS_NJOBS:-1000}"            # Total number of root files you want
EVENTS_PER_JOB="${CALOMAPS_NEVENTS:-20}"        # Keeps memory footprint low per file
CONCURRENT_JOBS="${CALOMAPS_NCONCURRENT:-20}"   # How many jobs to run simultaneously
# Job i is seeded SEED_BASE+i: every job unique, and the whole dataset reproducible.
# (bash $RANDOM is only 15-bit, so at ~1000 draws some collide and silently duplicate
# whole files.) Set CALOMAPS_SEED_BASE to generate a statistically independent dataset.
SEED_BASE="${CALOMAPS_SEED_BASE:-0}"

# --- PARTICLE / DATASET ---
# The particle type is read from CALOMAPS_GUN_PARTICLE (default "gamma") and passed
# through to run_sim.py. Defaults reproduce the original photon dataset exactly:
#   gamma -> data_spectrum_100um_400GeV/sim_photons_part*.root
# A non-photon run gets its own dataset dir + file prefix, e.g.:
#   CALOMAPS_GUN_PARTICLE=pi+ bash generate_batched.sh
#     -> data_spectrum_100um_400GeV_piplus/sim_piplus_part*.root
export CALOMAPS_GUN_PARTICLE="${CALOMAPS_GUN_PARTICLE:-gamma}"
case "$CALOMAPS_GUN_PARTICLE" in
    gamma) TAG="photons" ;;
    pi+)   TAG="piplus" ;;
    pi-)   TAG="piminus" ;;
    *)     TAG="$(echo "$CALOMAPS_GUN_PARTICLE" | sed 's/+/plus/g; s/-/minus/g')" ;;
esac
if [ "$CALOMAPS_GUN_PARTICLE" = "gamma" ]; then
    DATASET_NAME="${CALOMAPS_DATASET_NAME:-data_spectrum_100um_400GeV}"
else
    DATASET_NAME="${CALOMAPS_DATASET_NAME:-data_spectrum_100um_400GeV_${TAG}}"
fi
FILE_PREFIX="sim_${TAG}_part"
OUT_DIR="${CALOMAPS_DATA_BASE:-$HOME/CALOMAPS-data}/${DATASET_NAME}"

# Resolve the compact geometry + steering by ABSOLUTE path (relative to this script),
# so the script runs correctly from any working directory. SiD_TestBeam.xml lives in
# geometry/ and run_sim.py in sim/, so bare relative names can't both resolve.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
COMPACT="$REPO_DIR/geometry/SiD_TestBeam.xml"
STEER="$REPO_DIR/sim/run_sim.py"

# Refuse to run without ddsim -- BEFORE the destructive cleanup below, so an
# unsourced environment can't delete an existing dataset and then produce nothing.
if ! command -v ddsim >/dev/null 2>&1; then
    echo "ERROR: ddsim not found in PATH. Source setup/setup_calomaps.sh first." >&2
    exit 1
fi

echo "=========================================="
echo " Starting DD4hep Simulation Pipeline"
echo " Particle: $CALOMAPS_GUN_PARTICLE | Dataset: $DATASET_NAME"
echo " Total Jobs: $TOTAL_JOBS | Events/Job: $EVENTS_PER_JOB | Batch Size: $CONCURRENT_JOBS"
echo "=========================================="

# Create output directory safely
mkdir -p "$OUT_DIR"

# Clean up old simulation files to prevent mixing datasets
echo "-> Cleaning up old ROOT and log files in $OUT_DIR/..."
rm -f "$OUT_DIR/${FILE_PREFIX}"*.root
rm -f "$OUT_DIR"/log_job*.txt
rm -f "$OUT_DIR"/.fail_job* "$OUT_DIR/SIM_COMPLETE.txt"

# Counter to track active background jobs
active_jobs=0

for (( i=1; i<=TOTAL_JOBS; i++ ))
do
    echo "Submitting Job $i..."

    # A failed job leaves a .fail_job marker so the final status check can report it.
    ( ddsim --compactFile "$COMPACT" \
            --steeringFile "$STEER" \
            -N $EVENTS_PER_JOB \
            --random.seed $((SEED_BASE + i)) \
            --outputFile "$OUT_DIR/${FILE_PREFIX}${i}.root" \
            > "$OUT_DIR/log_job${i}.txt" 2>&1 \
      || touch "$OUT_DIR/.fail_job${i}" ) &

    # Increment our active job counter
    active_jobs=$((active_jobs + 1))

    # If we hit our concurrency limit, stop submitting and wait
    if [[ $active_jobs -eq $CONCURRENT_JOBS ]]; then
        echo ">>> Batch limit ($CONCURRENT_JOBS) reached. Waiting for batch to finish..."
        wait  # This pauses the script until all background jobs complete
        echo ">>> Batch complete! Moving to next set."
        active_jobs=0 # Reset counter for the next batch
    fi
done

# Catch any leftover jobs if TOTAL_JOBS isn't perfectly divisible by CONCURRENT_JOBS
echo ">>> Waiting for the final partial batch to finish..."
wait

# Only report success (and write the SIM_COMPLETE marker) if every job actually
# produced its output file and none left a failure marker.
n_failed=$(ls "$OUT_DIR"/.fail_job* 2>/dev/null | wc -l)
n_produced=$(ls "$OUT_DIR/${FILE_PREFIX}"*.root 2>/dev/null | wc -l)
if [ "$n_failed" -gt 0 ] || [ "$n_produced" -lt "$TOTAL_JOBS" ]; then
    echo "=========================================="
    echo " Simulation FINISHED WITH ERRORS:"
    echo "   failed jobs:  $n_failed  (see $OUT_DIR/.fail_job* and matching log_job*.txt)"
    echo "   output files: $n_produced of $TOTAL_JOBS expected"
    echo " NOT writing SIM_COMPLETE.txt."
    echo "=========================================="
    exit 1
fi

echo "=========================================="
echo " Simulation Complete! All $TOTAL_JOBS jobs produced output."
echo " Output: $OUT_DIR/${FILE_PREFIX}*.root"
echo "=========================================="

touch "$OUT_DIR/SIM_COMPLETE.txt"
