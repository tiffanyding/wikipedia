#!/bin/bash

months=("2018-05" "2018-06")
# months=("2018-01" "2018-02" "2018-03" "2018-04" "2018-05" "2018-06" 
#         "2018-07" "2018-08" "2018-09" "2018-10" "2018-11" "2018-12")

export TITLE_TO_IDX_FILE="data/wikilinkgraph/title_to_idx_2018.pkl"
    
for month in ${months[@]}
do
    export CLICKSTREAM_FILE="data/clickstream/clickstream-enwiki-${month}.tsv.gz"
    export SAVE_PREFIX="data/clickstream/cleaned/${month}"

    
    echo "Processing file $CLICKSTREAM_FILE"

    # # For debugging, uncomment the following lines and comment out 'qsub ...'
    # bash preprocessing/process_clickstream_file.sh

    # # Use -V flag to pass in environmental variables
    qsub -cwd -l day -V -l vf=50G preprocessing/process_clickstream_file.sh
 
done