#!/bin/bash

# Run this from wikipedia/ using `qsub -l day -cwd submit_job.sh`

source env/bin/activate
echo 'Running preprocessing/process_wikilinkgraph.py'
python preprocessing/process_wikilinkgraph.py
