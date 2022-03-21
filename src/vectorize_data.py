import argparse
import json
import os
import spacy
import sys

import numpy as np
import pandas as pd

vectorizer = spacy.load('en_core_web_md')
print('Got language model...')


def process_dir(dir_name:str):
    vec_container = []

    for fname in os.listdir(dir_name):
        with open(os.path.join(dir_name, fname), mode='r') as infile:
            sys.stdout.write('\rProcessing doc {}...'.format(fname))
            data = json.load(infile)
            if not('extracted_text' in data.keys()):
                continue

            text = data['extracted_text'].strip().replace('\n', '')
            vec_container.append(vectorizer(text).vector)

    return vec_container


def create_vectors(input_source:str):
    if not(input_source.endswith('jl') or input_source.endswith('csv')):
        print('Processing directory...')
        vec_container = process_dir(input_source)
    else:
        print('Reading in dataframe...')
        vec_container = []
        df = pd.read_json(input_source, lines=True)
        print('Starting vectorization...')
        for t_ix, t in enumerate(df.content.values):
            sys.stdout.write('\rGetting vector {}...'.format(t_ix))
            vec_container.append(vectorizer(t.strip().replace('\n', '')).vector)

    emb_matrix = np.asarray(vec_container)

    return emb_matrix


def main(args):
    emb_matrix = create_vectors(args.input_data)
    np.save(file=args.save_path, arr=emb_matrix)
    print('Embeddings saved!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_data', type=str, required=True)
    parser.add_argument('--save_path', type=str, required=True)
    args = parser.parse_args()

    main(args)