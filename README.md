# wikipedia

## Setup

## Setup virtual environment
Run `. create_venv.sh`

## Prepare WikiLinkGraphs data
1. Download the 2018 English WikiLinkGraph by running `bash scripts/download_wikilinkgraph.sh`. This downloads the data into `data/wikilinkgraph/`
1. Download a csv containing the titles of non-redirection Wikipedia pages as of 2020 by running `bash scripts/download_page_list.sh`. This list is used to filter out edges in WikiLinkGraph that are connected to redirection pages. 
1. Run `preprocessing/process_wikilinkgraph.py`. This can be submitted as a job on the CS grid by running `qsub -l day -cwd submit_job.sh`

## Create map from page index to page topic
1. Run `preprocessing/create_idx_to_topic_map.py`. This requires a good amount of memory (16gb is enough), so you may have to submit it as a job on the grid by modifying `submit_job.sh`. 

## Prepare Wikipedia clickstream data
[WIP. I will likely redo this section to incorporate the indices generated from processing WikiLinkGraphs]
1. Download the Wikipedia clickstream data by running `bash scripts/download_clickstream_data.sh`. This downloads the clickstream data for Jan 2018-Dec 2018 to `data/clickstream`. To modify which months are downloaded, you can edit the list of URLs in `data/wiki_clickstream_urls.txt`.
1. If you are on the CS grid, run `bash preprocessing/generate_clickstream_processing_script` and then run `bash preprocessing/process_all_clickstream_files.sh`. This submits one job per data file and saves processed filed to `data/clickstream/cleaned`.
1. Gather the outputs by running `python preprocessing/aggregate_processed_clickstreams.py`.

## Running analysis
The code for this is not very well organized...
- `random_walks.py` contains the bulk of the analysis, including computing correlation coefficients, fraction of users at the sink node over time, and topic distributions over time. This file calls on functions written in other files, including `compute_correlations.py` and `map_to_topic.py`
- `get_top_titles.py` can be run to get the top pages according to various metrics (PR, randomw walk visits, etc.)

## Note

I have not done a full run through of these instructions starting from a clean repo so it is possible that there are steps missing. I will do this at some point. 
