#!/bin/sh

export PYTHONPATH=$PYTHONPATH:`pwd`

cd "$(dirname "$0")"
source venv/bin/activate

cd eliotExtension
python3 main.py
