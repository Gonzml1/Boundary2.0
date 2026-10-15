#!/usr/bin/env bash
# Compila las DLL de codigos_cpp/ con MinGW-w64 (MSYS2 UCRT64).
# Requisitos (desde la terminal "MSYS2 UCRT64"):
#   pacman -S mingw-w64-ucrt-x86_64-gcc mingw-w64-ucrt-x86_64-gmp
# Uso:
#   ./build_dlls.sh            # compila todas
#   ./build_dlls.sh julia      # compila solo julia.dll
#
# Se linkea todo estático (libgomp, libstdc++, gmp) para que las DLL no dependan
# de libgomp-1.dll, libstdc++-6.dll, etc.
set -euo pipefail

cd "$(dirname "$0")/codigos_cpp"

CXX="${CXX:-g++}"
# -ffp-contract=off: sin fusionar multiplicación y suma (FMA), así la perturbación en CPU da
# exactamente lo mismo que en GPU (que compila con --fmad=false) y se pueden mezclar cuadros
FLAGS=(-O3 -march=native -ffp-contract=off -fopenmp -shared -static)

compilar() {
    local nombre="$1"; shift
    echo ">> $nombre.dll"
    "$CXX" "${FLAGS[@]}" -o "$nombre.dll" "$nombre.cpp" "$@"
}

objetivos=("${@:-mandelbrot julia burning_ship circulo perturbacion gmp_puro}")
for nombre in ${objetivos[@]}; do
    case "$nombre" in
        perturbacion|gmp_puro) compilar "$nombre" -lgmp ;;
        *)                     compilar "$nombre" ;;
    esac
done
echo "Listo."
