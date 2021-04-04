import os
import pathlib

# Run this from within the preprocessing/ directory

# Generate script
script_file = 'process_clickstream_file.sh'

wikipedia_abs_path = pathlib.Path(os.path.abspath(__file__)).parents[1]
script = ('#!/bin/bash\n'
    f'cd {wikipedia_abs_path}\n'
    'source env/bin/activate\n'
    f'cd {os.getcwd()}\n'
    f'python process_clickstream.py'
    f' ${CLICKSTREAM_FILE} ${TITLE_TO_IDX_FILE} ${SAVE_PREFIX} \n')

with open(script_file, 'w') as f:
    f.write(script)
print(f'Wrote script to {script_file}.')