#!/bin/bash

cd /gpfs/main/home/tding5/workspace/wikipedia
source env/bin/activate
cd /gpfs/main/home/tding5/workspace/wikipedia
python preprocessing/process_clickstream.py $FILE_NAME $SAVE_PREFIX
