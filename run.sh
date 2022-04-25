#!/bin/sh

export PYTHONPATH=$PYTHONPATH:`pwd`

cd "$(dirname "$0")"

cd eliotExtension
python3 main.py
