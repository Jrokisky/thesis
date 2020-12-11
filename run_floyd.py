# Script for simplifying running floydhub cli commands.
# This command needed to be run 60+ times.

import argparse
import subprocess

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('obj')
    parser.add_argument('search_term')
    args = parser.parse_args()
    search_term = args.search_term
    obj = args.obj

    cmd = f'python binary.py {obj} "{search_term}"'
    print(cmd)

    subprocess.call(['floyd', 'run', cmd, '-m', obj])


if __name__ == '__main__':
    main()
