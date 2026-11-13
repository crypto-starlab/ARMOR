"""Resolve the shared training modules and this checkout's native HE library."""
import ctypes
from pathlib import Path
import sys

ARMOR_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ARMOR_ROOT))
NATIVE = ARMOR_ROOT / 'phantom-fhe' / 'build' / 'lib'
if (NATIVE / 'libPhantom.so').exists():
    ctypes.CDLL(str(NATIVE / 'libPhantom.so'), mode=ctypes.RTLD_GLOBAL)
sys.path.insert(0, str(NATIVE))
