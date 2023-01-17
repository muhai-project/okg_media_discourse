#!/bin/sh

# recursively removes all .pyc files and __pycache__ directories in the current
# directory

find . | \
  grep -E "(__pycache__|\.pyc$)|.ipynb_checkpoints|htmlcov|.coverage" | \
  xargs rm -rf