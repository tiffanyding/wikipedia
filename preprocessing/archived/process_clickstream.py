import argparse
import pandas as pd
import pickle
import time

st = time.time()

def process_clickstream_file(zip_file, save_prefix=None, dct={}):
    print('File: ', zip_file)
    df = pd.read_csv(zip_file, compression='gzip', sep='\t',
                # nrows=10000, # Can uncomment when debugging
                names=['prev', 'curr', 'type', 'n'])
    num_rows = len(df)
    print('Number of rows:', num_rows)

    # Filter out rows of type other (correspond to non-existent edges)
    filtered_df = df[~(df['type']=='other')]
    print('Number of rows with type != other:', len(filtered_df))

    for i, row in df.iterrows():
        if i % 100000 == 0:
            print(f'Processed {i} out of {num_rows} rows')
        if row['type'] == 'external':
            start_vertex = 'external'
        else: # type is link
            start_vertex = row['prev']
        
        end_vertex = row['curr']
        num_clicks = row['n']

        # Add count to start_vertex out_edges dict (increment count if it already exists)
        if start_vertex not in dct:
            dct[start_vertex] = {'in_edges': {}, 'out_edges': {}}
        dct[start_vertex]['out_edges'][end_vertex] = dct[start_vertex]['out_edges'].get(end_vertex, 0) + num_clicks

        # Add count to end_vertex in_edges dict
        if end_vertex not in dct:
            dct[end_vertex] = {'in_edges': {}, 'out_edges': {}}
        dct[end_vertex]['in_edges'][start_vertex] = dct[end_vertex]['in_edges'].get(start_vertex, 0) + num_clicks

    pages = set(list(filtered_df['prev']) + list(filtered_df['curr']))

    # Save dct and pages
    if save_prefix is not None:
        dct_file_name = f'{save_prefix}dict.pkl'
        pages_file_name = f'{save_prefix}pages.pkl'
        
        with open(dct_file_name, 'wb') as f:
            pickle.dump(dct, f)
        print(f'Saved clickstream dict to {dct_file_name}')
        with open(pages_file_name, 'wb') as f:
            pickle.dump(pages, f)
        print(f'Saved list of pages to {pages_file_name}')

    return dct, pages

# # As a sanity check, print first n elements from clickstream_dict
# n = 10
# j = 0
# for k, v in clickstream_dict.items():
#     print(k, v)
#     print()
#     j += 1
#     if j > n:
#         break

# print(pages)


if __name__ == '__main__':
    '''
    Example usage: Run this from wikipedia/ (Assumes that 'data/clickstream/cleaned/' already exists)
        python preprocessing/process_clickstream.py 'data/clickstream/clickstream-enwiki-2018-01.tsv.gz' 'data/clickstream/cleaned/2018-01'
    '''
    parser = argparse.ArgumentParser(
        description="Process a Wikipedia clickstream file"
    )
    parser.add_argument("file", type=str, help="path to gzipped Wikipedia clickstream file")
    parser.add_argument("save_prefix", type=str, help="output will be saved to <save_prefix>dict.pkl and <save_prefix>pages.pkl")
    args = parser.parse_args()

    st = time.time()

    process_clickstream_file(args.file, save_prefix=args.save_prefix, dct={})

    print(f'Time taken: {(time.time() - st) / 60:.2f} min')