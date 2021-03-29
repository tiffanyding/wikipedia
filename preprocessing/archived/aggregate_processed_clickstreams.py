import glob, os
import pandas as pd
import pickle

folder = "data/clickstream/cleaned"

# -------------------------

def load_pickle_file(path):
    with open(path, 'rb') as f:
        loaded_file = pickle.load(f)

    return loaded_file

# Get lists of dict file names and pages file names
dict_files = []
pages_files = []
for root, dirs, files in os.walk(folder):
    for f in files:
        if f.endswith('dict.pkl'):
            dict_files.append(f)
        elif f.endswith('pages.pkl'):
            pages_files.append(f)
dict_files = sorted(dict_files)
pages_files = sorted(pages_files)

# Aggregate pages by taking the union 
# combined_pages = set()
# for pages_path in pages_files:
#     pages = load_pickle_file(os.path.join(folder, pages_path))
#     print(f'Page count for "{pages_path}": {len(pages)}')
#     combined_pages = set(list(combined_pages) + list(pages))
# print('Total number of unique pages:', len(combined_pages))

# # Save aggregated pages
# save_to = os.path.join(folder, 'aggregated_pages.pkl')
# with open(save_to, 'wb') as f:
#     pickle.dump(combined_pages, f)
# print(f'Saved aggregated pages to {save_to}')

# Aggregate dicts by adding together counts
print(f'Processing {dict_files[0]}')
combined_dict = load_pickle_file(os.path.join(folder, dict_files[0]))

for i in range(1, len(dict_files)):
    print(f'Processing {dict_files[i]}')
    dct = load_pickle_file(os.path.join(folder, dict_files[i]))
    for start_vertex, curr_dict in dct.items():

        if start_vertex not in combined_dict:
            combined_dict[start_vertex] = {'in_edges': {}, 'out_edges': {}}

        for end_vertex, num_clicks in curr_dict.items():
            combined_dict[start_vertex]['out_edges'][end_vertex] = \
                combined_dict[start_vertex]['out_edges'].get(end_vertex, 0) + num_clicks

# Save aggregated dict
save_to = os.path.join(folder, 'aggregated_dicts.pkl')
with open(save_to, 'wb') as f:
    pickle.dump(combined_dicts, f)
print(f'Saved aggregated dicts to {save_to}')




            