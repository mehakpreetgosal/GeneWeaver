"""
pytest setup. If this machine has no NVIDIA driver, switch Numba to its CPU CUDA simulator so the
GPU tests still run (slowly) and check the kernel LOGIC. This must happen before numba is imported.
Force it either way with  GENEWEAVER_SIM=1  (simulator)  or  GENEWEAVER_SIM=0  (real GPU).
"""
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_force = os.environ.get("GENEWEAVER_SIM")
if _force == "1" or (_force is None and shutil.which("nvidia-smi") is None):
    os.environ["NUMBA_ENABLE_CUDASIM"] = "1"
