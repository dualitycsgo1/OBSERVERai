"""Run the automated regression suite; match accuracy needs separate demo evaluation."""
from pathlib import Path
import subprocess
import sys


def main():
    return subprocess.call([sys.executable, '-m', 'pytest', 'tests',
                            'test_demo_processing.py', '-q'], cwd=Path(__file__).resolve().parent)


if __name__ == '__main__':
    raise SystemExit(main())
