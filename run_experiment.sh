#!/usr/bin/env bash
set -euo pipefail  # robustere Einstellungen

# --- Argumente einlesen -------------------------------------------------------
# Aufruf z.B.: ./run_experiment.sh SSALS 1 0
#              ./run_experiment.sh <ALS> <radius_option> <seed>

if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <ALS> <radius_option> <seed>"
    echo "z.B.:  $0 SSALS 1 0"
    exit 1
fi

ALS="$1"
RADIUS_OPTION="$2"
SEED="$3"

# --- Experiment-Verzeichnis aus Argumenten bauen ------------------------------
# Hier kannst du das Schema anpassen, wie du magst
EXP_DIR="experiments0/${ALS}_r${RADIUS_OPTION}"
EXP_NAME="seed${SEED}"
EXP_DIR="${EXP_DIR}/${EXP_NAME}"

echo "Experiment-Verzeichnis: ${EXP_DIR}"
mkdir -p "${EXP_DIR}"

# 1) Experiment initialisieren
python initialize_experiment.py "${EXP_DIR}" \
    --radius_option "${RADIUS_OPTION}" \
    --als "${ALS}" \
    --seed "${SEED}"

# 2) FEM-Daten erzeugen: l = 6 ... 10
for l in {6..10}; do
    echo "Running: python generate_femdata.py ${EXP_DIR} --l ${l}"
    python generate_femdata.py "${EXP_DIR}" --l "${l}"
done

# 3) Surrogates ohne refine: l = 6 ... 10
for l in {6..10}; do
    echo "Running: python compute_surrogates.py ${EXP_DIR} --l ${l}"
    python compute_surrogates.py "${EXP_DIR}" --l "${l}"
done

# 4) Surrogates mit refine: l = 7 ... 10
for l in {7..10}; do
    echo "Running: python compute_surrogates.py ${EXP_DIR} --l ${l} --refine"
    python compute_surrogates.py "${EXP_DIR}" --l "${l}" --refine
done

# for als in SALS SSALS; do for r in 1 2; do ./run_experiment.sh "$als" "$r" 9; done; done