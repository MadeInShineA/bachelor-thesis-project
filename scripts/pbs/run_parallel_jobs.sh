#!/bin/bash
# Please note that this file was adapted from one given by Ayumu-san
# The argument parsing section was taken from the ../parallel_jobs.sh file, which was written by Qwen3.6-Plus

source /home/cbi/olivier.amacker/bachelor-thesis/bachelor-thesis-project/scripts/pbs/config.sh

# Initialize all variables to empty
RUN_NUM=""
SUBJECTS_STR=""
DATA_DIR=""
N_SUBS_IN_PARALLEL=""
OUT_DIR=""
WORK_DIR=""
LICENSE_FILE=""

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --run-num)
      [[ -z "${2:-}" ]] && { echo "Error: --run-num requires a value"; exit 1; }
      RUN_NUM="$2"; shift 2 ;;
    --subjects)
      [[ -z "${2:-}" ]] && { echo "Error: --subjects requires a value"; exit 1; }
      SUBJECTS_STR="$2"; shift 2 ;;
    --data-dir)
      [[ -z "${2:-}" ]] && { echo "Error: --data-dir requires a value"; exit 1; }
      DATA_DIR="$2"; shift 2 ;;
    --n-subs-in-parallel)
      [[ -z "${2:-}" ]] && { echo "Error: --n-subs-in-parallel requires a value"; exit 1; }
      N_SUBS_IN_PARALLEL="$2"; shift 2 ;;
    --out-dir)
      [[ -z "${2:-}" ]] && { echo "Error: --out-dir requires a value"; exit 1; }
      OUT_DIR="$2"; shift 2 ;;
    --work-dir)
      [[ -z "${2:-}" ]] && { echo "Error: --work-dir requires a value"; exit 1; }
      WORK_DIR="$2"; shift 2 ;;
    --license-file)
      [[ -z "${2:-}" ]] && { echo "Error: --license-file requires a value"; exit 1; }
      LICENSE_FILE="$2"; shift 2 ;;
    *)
      echo "Unknown option: $1"
      exit 1 ;;
  esac
done

# Validate all required arguments are provided
if [[ -z "$RUN_NUM" || -z "$SUBJECTS_STR" || -z "$DATA_DIR" || -z "$N_SUBS_IN_PARALLEL" || -z "$OUT_DIR" || -z "$WORK_DIR" || -z "$LICENSE_FILE" ]]; then
  cat <<EOF
Error: All arguments are required. No defaults are set.

Usage:
  $0 --run-num <num> \
     --subjects <sub1,sub2,...> \
     --data-dir <path> \
     --n-subs-in-parallel <int> \
     --out-dir <path> \
     --work-dir <path> \
     --license-file <path>

Arguments:
  --run-num                     Run identifier (e.g., 1)
  --subjects                    Comma-separated subject IDs (e.g., "001,060,120")
  --data-dir                    Path to BIDS data directory
  --n-subs-in-parallel          Number of parallel subs Docker jobs to run
  --out-dir                     Path to output directory
  --work-dir                    Path to work/scratch directory
  --license-file                Path to FreeSurfer license file
EOF
  exit 1
fi

# Validate N_SUBS_IN_PARALLEL is a positive integer
if ! [[ "$N_SUBS_IN_PARALLEL" =~ ^[0-9]+$ ]] || [[ "$N_SUBS_IN_PARALLEL" -eq 0 ]]; then
  echo "Error: --n-subs-in-parallel must be a positive integer greater than 0."
  exit 1
fi

# Validate license file exists
if [[ ! -f "$LICENSE_FILE" ]]; then
  echo "Error: License file not found at $LICENSE_FILE"
  exit 1
fi

# Convert comma-separated subjects to bash array
IFS=',' read -r -a SUBJECTS <<< "$SUBJECTS_STR"
if [[ ${#SUBJECTS[@]} -eq 0 ]]; then
  echo "Error: No subjects provided."
  exit 1
fi

function find_usenode {
  BEST_USAGE=1.0  # 使用率初期値 (100%)

  for i in "${NODE[@]}"; do
    COUNT=$(qstat -nr | grep -c "${NODENAME}${i}")
    CORES=${NODE_CORES[$i]}
    USAGE=$(echo "scale=4; ${COUNT}/${CORES}" | bc)
	echo "Node: ${NODENAME}${i}, Usage: ${USAGE}, Cores: ${CORES}, Count: ${COUNT}"
    if (( $(echo "$USAGE < $BEST_USAGE" | bc -l) )); then
      BEST_USAGE=$USAGE
      use_node=$i
    fi
  done
}

echo "Now job throw"

while true; do
    current_job_num=$(qstat | grep ${username} | wc -l)
    find_usenode

    COUNT=$(qstat -nr | grep -c "${NODENAME}${use_node}")
    CORES=${NODE_CORES[$use_node]}

    if [ $(( COUNT + N_SUBS_IN_PARALLEL )) -le "$CORES" ]; then
      break
    else
      echo "* waiting for available resources *"
      sleep 0.2
    fi
done

# Create a temporary job script
TMP_JOB="/home/cbi/olivier.amacker/bachelor-thesis/bachelor-thesis-project/scripts/pbs/temp_sub.sh"


echo "#!/bin/bash
    /home/cbi/olivier.amacker/bachelor-thesis/bachelor-thesis-project/scripts/parallel_jobs.sh \
        --n-subs-in-parallel "${N_SUBS_IN_PARALLEL}" \
        --run-num "${RUN_NUM}" \
        --subjects "${SUBJECTS_STR}" \
        --data-dir "${DATA_DIR}" \
        --out-dir "${OUT_DIR}" \
        --work-dir "${WORK_DIR}" \
        --license-file "${LICENSE_FILE}"
    " > ${TMP_JOB}

chmod +x "$TMP_JOB"

# Create the job name
SAFE_SUBJECTS=$(echo "${SUBJECTS_STR}" | tr ',' '-')
SUBJECT_HASH=$(echo -n "${SAFE_SUBJECTS}" | md5sum | cut -c1-8)
JOB_NAME="fuzzy-fmriprep-run${RUN_NUM}-${SUBJECT_HASH}"

# Create a metadata file for the job
META_FILE="./jobs-metadata/job_${SAFE_SUBJECTS}_run_${RUN_NUM}_metadata.txt"

echo "PBS_JOB_NAME: ${JOB_NAME}
    SUBJECTS: ${SAFE_SUBJECTS}
    RUN_NUMBER: ${RUN_NUM}
    N_SUBS_IN_PARALLEL: ${N_SUBS_IN_PARALLEL}
    SUBMIT_TIME: $(date)
    SUBMITTER: $(whoami)
    " > ${META_FILE}

JOB_ID=$(qsub -N ${JOB_NAME} -l nodes="${NODENAME}${use_node}:ppn=${N_SUBS_IN_PARALLEL}" "$TMP_JOB")

rm -f "$TMP_JOB"

echo "Submitted job ${JOB_ID} to ${NODENAME}${use_node}"
sleep 1

echo "done!!"
