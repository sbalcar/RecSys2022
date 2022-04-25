#!/bin/sh

#cd "$(dirname "$0")"


#export PYTHONPATH=$PYTHONPATH:/home/stepan/workspaceJup/elliot
export PYTHONPATH=$PYTHONPATH:`pwd`

cd "$(dirname "$0")"

cd eliotExtension
python3 main.py -generateDatasets
