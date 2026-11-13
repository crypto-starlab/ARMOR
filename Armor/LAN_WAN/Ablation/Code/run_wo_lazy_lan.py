#!/usr/bin/env python3
"""Compatibility entry point for the shared w/o Lazy LAN launcher."""
import sys
from run_benchmark_lan import main
if __name__=='__main__':sys.exit(main(['--method','wo_Lazy',*sys.argv[1:]]))
