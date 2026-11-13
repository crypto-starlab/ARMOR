#!/usr/bin/env python3
"""Run wo_Coeff with the shared Fig4 LAN experiment configuration."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'Code'))
from run_benchmark_lan import main
if __name__=='__main__':
    sys.exit(main(['--method','wo_Coeff',*sys.argv[1:]]))
