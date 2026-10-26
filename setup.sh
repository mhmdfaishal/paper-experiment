#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python3 -m venv .venv
.venv/bin/python -m pip install -r ga_pso/requirements.txt
Rscript -e 'if (!requireNamespace("Rcpp", quietly=TRUE)) install.packages("Rcpp", repos="https://cloud.r-project.org")'
bash its_mcttrp/src/build.sh
