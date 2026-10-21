#!/usr/bin/env bash
# Compila las bibliotecas de C++ que usa el render (perturbación y GMP) para Linux.
# Quedan en codigos_cpp/ como .so, al lado de las .dll de Windows; core/backend_cpp.py las carga solo.
# Requisitos: g++ y libgmp-dev (los instala instalar.sh).
set -euo pipefail

cd "$(dirname "$0")/../codigos_cpp"

CXX="${CXX:-g++}"
# -ffp-contract=off: sin FMA, igual que en Windows y que la GPU (--fmad=false), así CPU y GPU dan
# exactamente lo mismo. -march=native: se compila en la misma máquina que va a calcular.
FLAGS=(-O3 -march=native -ffp-contract=off -fopenmp -shared -fPIC)

for nombre in perturbacion gmp_puro; do
    echo ">> codigos_cpp/$nombre.so"
    "$CXX" "${FLAGS[@]}" -o "$nombre.so" "$nombre.cpp" -lgmp
done
echo "Listo."
