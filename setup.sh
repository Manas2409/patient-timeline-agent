#!/usr/bin/env bash
# Setup script for "Unified Patient History & Smart Visit Agent".
# Works in Git Bash (Windows), macOS, and Linux.
#
# What this does:
#   1. Creates a Python virtual environment and installs dependencies.
#   2. Creates .env from .env.example if it doesn't exist yet, and pauses for you
#      to fill in Neo4j + OpenAI credentials (both free to obtain -- see below).
#   3. Downloads the raw Synthea sample dataset (not committed to the repo).
#   4. Runs the data-generation pipeline (scripts 01-06; no external services needed).
#   5. Loads the result into Neo4j (script 07; needs the credentials from step 2).
#   6. Prints ready-to-run sample commands for the CLI and the Streamlit app.
#
# Run it from the project root: ./setup.sh

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

echo "=== Unified Patient History & Smart Visit Agent -- setup ==="
echo

# --- 1. Virtual environment -------------------------------------------------
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment (.venv)..."
    if command -v python3 >/dev/null 2>&1; then
        python3 -m venv .venv
    else
        python -m venv .venv
    fi
fi

if [ -f ".venv/Scripts/activate" ]; then
    # shellcheck disable=SC1091
    source .venv/Scripts/activate   # Windows (venv created with Windows Python)
else
    # shellcheck disable=SC1091
    source .venv/bin/activate       # macOS/Linux
fi

echo "Installing dependencies (this downloads torch/transformers -- can take a few minutes)..."
pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt
echo "Dependencies installed."
echo

# --- 2. Environment variables -----------------------------------------------
if [ ! -f ".env" ]; then
    cp .env.example .env
    echo "Created .env from .env.example."
fi

if ! grep -q "NEO4J_URI=neo4j" .env 2>/dev/null; then
    cat <<'EOF'

Before continuing, fill in .env with:
  - NEO4J_URI / NEO4J_USERNAME / NEO4J_PASSWORD / NEO4J_DATABASE
      Free instance: https://console.neo4j.io -> New Instance -> AuraDB Free
      (download the generated credentials when it offers them -- shown only once)
  - OPENAI_API_KEY
      https://platform.openai.com/api-keys (needs a small amount of credit)

Steps 1-6 below don't need either of these and will run regardless; step 7
(loading Neo4j) and the agent/Streamlit app do need them.

EOF
    read -rp "Press Enter once .env is filled in (or Ctrl+C to stop here and re-run this script later)... "
fi

# --- 3. Raw Synthea sample data (not committed -- large, third-party) -------
RAW_DIR="data/raw"
if [ ! -f "$RAW_DIR/patients.csv" ]; then
    echo
    echo "=== Downloading Synthea sample data (~6MB) ==="
    mkdir -p "$RAW_DIR"
    SYNTHEA_URL="https://synthetichealth.github.io/synthea-sample-data/downloads/latest/synthea_sample_data_csv_latest.zip"
    if curl -fsSL -o "$RAW_DIR/synthea_sample.zip" "$SYNTHEA_URL"; then
        (cd "$RAW_DIR" && unzip -oq synthea_sample.zip && rm synthea_sample.zip)
        echo "Downloaded and extracted to $RAW_DIR/."
    else
        cat <<EOF

Could not download automatically. Please download by hand:
  $SYNTHEA_URL
and extract its CSVs directly into: $RAW_DIR/
Then re-run this script.
EOF
        exit 1
    fi
fi

# --- 4. Data-generation pipeline (no external services needed) -------------
echo
echo "=== Running the data pipeline (steps 1-6) ==="
python scripts/01_shatter_and_corrupt.py
python scripts/02_generate_notes.py
python scripts/03_entity_resolution.py
echo "(Step 4 downloads a ~400MB biomedical NER model on first run -- this can take a while.)"
python scripts/04_extract_entities.py
python scripts/05_normalize_entities.py
python scripts/06_merge_timeline.py

# --- 5. Load into Neo4j ------------------------------------------------------
echo
echo "=== Loading the unified timelines into Neo4j (step 7) ==="
python scripts/07_load_neo4j.py

# --- 6. Ready to go -----------------------------------------------------------
cat <<'EOF'

=== Setup complete ===

Try the CLI (reproducible test patient, "Thuy VonRueden"):
  python scripts/10_lookup_patient.py --first Thuy --last VonRueden --dob 1998-07-22 --gender F
  python scripts/09_patient_history_view.py --patient 31 \
      --complaint "fever and sore throat, suspected bacterial infection" \
      --treatment "considering amoxicillin"

Or launch the web app:
  streamlit run streamlit_app.py

See README.md for the full pipeline explanation, honest accuracy numbers, and
known limitations.
EOF
