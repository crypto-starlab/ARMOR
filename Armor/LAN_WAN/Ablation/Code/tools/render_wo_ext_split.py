#!/usr/bin/env python3
"""Render wo_Ext using the common Fig4 result-table format."""
import sys
from render_results import main
if __name__=='__main__':
    sys.argv[1:1]=['--method','wo_Ext']
    main()
