"""wo_Lazy norm benchmark: server1 entry point."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from norm_benchmark import main


if __name__ == '__main__':
    main('wo_Lazy', 'server1')
