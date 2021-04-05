import argparse
import pandas as pd
import pickle
import scipy.sparse as ss
import time

st = time.time()

def process_clickstream_file(zip_file, title_to_idx_file, save_prefix=None):
    '''
    Inputs:
        zip_file: Path to gzipped file containing Wikipedia clickstream data 
        title_to_idx_file: Path to pickle file generated by process_wikilinkgrapy.py
    '''
    # Read in files
    print('File: ', zip_file)
    df = pd.read_csv(zip_file, compression='gzip', sep='\t',
                # nrows=10000, # Can uncomment when debugging
                names=['prev', 'curr', 'type', 'n'])
    num_rows = len(df)
    print('Number of rows:', num_rows)

    with open(title_to_idx_file, 'rb') as f:
        title_to_idx = pickle.load(f)
    num_pages = len(title_to_idx)

    # print(df.head())
    # Convert underscores in page titles to spaces
    print('Converting underscores in page titles to spaces...')
    df['prev'] = df['prev'].apply(lambda x: str(x).replace('_', ' '))
    df['curr'] = df['curr'].apply(lambda x: str(x).replace('_', ' '))
    print(df.head())

    # Filter out rows of type other (correspond to non-existent edges)
    df = df[~(df['type']=='other')]
    print('Number of rows with type != other:', len(df))
    print('Number of rows with type = external:', len(df[df['type']=='external']))
    print('Number of rows with type = link:', len(df[df['type']=='link']))

    # We will add an extra row to account for links from external vertices  
    external_page_idx = num_pages

    # There may be a mismatch between the pages that exist in WikiLinkGraphs and the 
    # clickstream files, so we exclude rows that are marked 'internal' but one or both
    # of the pages are not in title_to_idx
    num_rows_before = len(df)
    df = df[~((df['type'] == 'link') & (~(df['prev'].isin(title_to_idx)) | ~(df['curr'].isin(title_to_idx))))]
    print(f'Excluded {num_rows_before - len(df)} rows that include pages not present in WikiLinkGraphs')

    # Map page titles to indices
    df['start_idx'] = df['prev'].apply(lambda x: title_to_idx.get(x, external_page_idx)) # Maps external pages to external_page_idx
    df['end_idx'] = df['curr'].apply(lambda x: title_to_idx[x])

    rows = df['start_idx']  # Not a copy, just a reference.
    cols = df['end_idx']
    data = df['n']
    matrix = ss.coo_matrix((data, (rows, cols)), shape=(num_pages+1, num_pages))
    matrix = ss.csc_matrix(matrix)

    # Save matrix
    if save_prefix is not None:
        file_name = f'{save_prefix}_matrix.pkl'
        
        with open(file_name, 'wb') as f:
            pickle.dump(matrix, f)
        print(f'Saved clickstream matrix to {file_name}')

    return matrix

if __name__ == '__main__':
    '''
    Example usage: Run this from wikipedia/ (Assumes that 'data/clickstream/cleaned/' already exists)
        python preprocessing/process_clickstream.py 'data/clickstream/clickstream-enwiki-2018-01.tsv.gz' 'data/wikilinkgraph/title_to_idx_map_2002.pkl' 'data/clickstream/cleaned/2018-01'
    '''
    parser = argparse.ArgumentParser(
        description="Process a Wikipedia clickstream file"
    )
    parser.add_argument("clickstream_file", type=str, help="path to gzipped Wikipedia clickstream file")
    parser.add_argument("title_to_idx_file", type=str, help="path to pickle file map from page titles to index," 
                        "as generated by preprocessing/process_wikilinkgraphs.py")
    parser.add_argument("save_prefix", type=str, help="output will be saved to <save_prefix>matrix.pkl")
    args = parser.parse_args()

    st = time.time()

    process_clickstream_file(args.clickstream_file, args.title_to_idx_file, save_prefix=args.save_prefix)

    print(f'Time taken: {(time.time() - st) / 60:.2f} min')