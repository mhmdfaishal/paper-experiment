#!/usr/bin/env bash
# Build the Rcpp move-list library as itsmcttrp.so (replaces the Windows-only .dll).
set -euo pipefail
cd "$(dirname "$0")"
export PKG_CPPFLAGS="$(Rscript -e 'Rcpp:::CxxFlags()')"
export PKG_LIBS=""   # Rcpp needs no extra link flags
R CMD SHLIB -o itsmcttrp.so movlist.cpp
