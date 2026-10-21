#!/usr/bin/env bash
# Prepara una máquina Linux con GPU NVIDIA (por ejemplo una alquilada en vast.ai o RunPod) para
# correr render_video.py: paquetes del sistema, Python, CuPy, compilación y una comprobación final.
#
#   bash render_nube/instalar.sh
#
# Pensado para Ubuntu/Debian con el driver de NVIDIA ya instalado (las imágenes "CUDA" o "PyTorch"
# de los proveedores lo traen). Se puede correr más de una vez.
set -euo pipefail

RAIZ="$(cd "$(dirname "$0")/.." && pwd)"
cd "$RAIZ"

SUDO=""
if [ "$(id -u)" -ne 0 ]; then SUDO="sudo"; fi

echo "== Paquetes del sistema (g++, GMP, ffmpeg, Python)"
$SUDO apt-get update -qq
$SUDO env DEBIAN_FRONTEND=noninteractive apt-get install -y -qq g++ libgmp-dev ffmpeg python3 python3-venv python3-pip >/dev/null

echo "== Entorno de Python (render_nube/.venv)"
python3 -m venv render_nube/.venv
# shellcheck disable=SC1091
source render_nube/.venv/bin/activate
pip install -q --upgrade pip
pip install -q -r render_nube/requirements.txt

# CuPy: el paquete depende de la versión de CUDA del driver (nvidia-smi la muestra)
CUDA_MAYOR="$(nvidia-smi 2>/dev/null | sed -n 's/.*CUDA Version: *\([0-9]*\).*/\1/p' | head -1)"
if [ -z "$CUDA_MAYOR" ]; then
    echo "No encontré la GPU (nvidia-smi no responde): el render va a usar solo la CPU." >&2
else
    if [ "$CUDA_MAYOR" -ge 13 ]; then PAQUETE="cupy-cuda13x"; else PAQUETE="cupy-cuda12x"; fi
    echo "== CuPy para CUDA $CUDA_MAYOR ($PAQUETE, con sus bibliotecas de CUDA)"
    pip install -q "$PAQUETE[ctk]" || pip install -q "$PAQUETE"
fi

echo "== Compilando las bibliotecas de C++"
bash render_nube/compilar_linux.sh

echo "== Comprobación"
python - <<'EOF'
import sys
sys.path.insert(0, ".")
from core.modulo_de_calculo_fractales import BACKENDS_DISPONIBLES, FRACTAL_REGISTRY
print("Bibliotecas:", {k: v for k, v in BACKENDS_DISPONIBLES.items()})
print("Métodos de Mandelbrot:", ", ".join(FRACTAL_REGISTRY.get("Mandelbrot", {})))
if BACKENDS_DISPONIBLES.get("gpu"):
    import cupy as cp
    from core.funciones_kernel import kernel_perturbacion
    for i in range(cp.cuda.runtime.getDeviceCount()):
        with cp.cuda.Device(i):
            kernel_perturbacion("mandelbrot", 2)          # compila el kernel en cada placa
            print(f"GPU {i}: {cp.cuda.runtime.getDeviceProperties(i)['name'].decode()} lista")
if not BACKENDS_DISPONIBLES.get("perturbacion.dll"):
    sys.exit("Falta la biblioteca de perturbación: revisá los errores de la compilación.")
EOF

echo
echo "Listo. Para renderizar:"
echo "  source render_nube/.venv/bin/activate"
echo "  python render_nube/render_video.py CARPETA_DEL_VIDEO"
