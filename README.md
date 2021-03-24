# wikipedia

## Setup virtual environment
Run `. create_venv.sh`

## Prepare WikiLinkGraphs data
1. Download the 2018 English WikiLinkGraph by running `. scripts/download_wikilinkgraph.sh`. This downloads the data into `data/wikilinkgraph/`
1. Run `preprocessing/process_wikilinkgraph.py`. This can be submitted as a job on the CS grid by running `qsub -l day -cwd submit_job.sh`

## Prepare Wikipedia clickstream data
[WIP. I will likely redo this section to incorporate to indices generated from processing WikiLinkGraphs]
1. Download the Wikipedia clickstream data by running `. scripts/download_clickstream_data.sh`. This downloads the clickstream data for Jan 2018-Dec 2018 to `data/clickstream`. To modify which months are downloaded, you can edit the list of URLs in `data/wiki_clickstream_urls.txt`.
1. If you are on the CS grid, run `. preprocessing/generate_clickstream_processing_script` and then run `. preprocessing/process_all_clickstream_files.sh`. This submits one job per data file and saves processed filed to `data/clickstream/cleaned`.
<!-- 1. Gather the outputs by running  -->
