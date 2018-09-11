import pandas as pd
import os
import time
import numpy as np

def files(path):
    '''Gets the filenames from a directory'''
    for file in os.listdir(path):
        if os.path.isfile(os.path.join(path, file)):
            yield file


def spread_artID(df):
    '''Associates each sentence with its corresponding article ID'''
    artID_list = df['art_ID'].tolist()
    artID = 0
    for idx, elem in enumerate(artID_list):
        if not np.isnan(elem):
            artID = elem
        else:
            artID_list[idx] = artID
    df['art_ID'] = artID_list

    return df


def get_word(idx, word_mapping):
    return word_mapping.iloc[idx, word_mapping.columns.get_loc('word')]


def create_docs_df(dir_path, out_path):
    '''Reads the WiNER's documents and creates the associated dataframes.'''

    word_mapping = pd.read_csv('./corpus_WiNER/document.vocab', sep=' ', header=None, 
                           names=['word', 'frequency'], keep_default_na=False)

    docs = pd.DataFrame(columns=['sentence','art_ID'])
    numOfSentences = 0
    count = 0 # Used for the 'progress bar'

    for file in files(dir_path):
        filepath = dir_path + file
        df_path = out_path + 'doc_' + file

        doc = pd.read_csv(filepath, sep='ID', engine='python', header=None,
                          names=['sentence', 'art_ID'])
        doc = spread_artID(doc)
        # Remove the rows with NaN that contained the IDs of the articles initially.
        doc = doc.dropna()
        doc['sentence'] = doc['sentence'].map(lambda x: list(map(int, x.split(' '))))
        # Mapping
        doc['sentence'] = doc['sentence'].map(lambda sen: [get_word(idx, word_mapping) for idx in sen])
        numOfSentences += doc.shape[0]
        if count % 10 == 0:
            print(count, end=' ')
        count += 1
        
        pd.to_pickle(doc, df_path)
    print('Cantidad de oraciones: {}'.format(numOfSentences))

    
def get_tag(x):
    tags = ['PER', 'LOC', 'ORG', 'MISC']
    return tags[x]    
    
def create_coarseNE_df(dir_path, out_path):
    '''Reads the WiNER's coarseNE files and creates the associated dataframes.'''

    numOfEntities = 0
    count = 0
    for file in files(dir_path):
        filepath = dir_path + file
        df_path = out_path + 'coarseNE_' + file

        coarseNE = pd.read_csv(filepath, sep='ID', engine='python', header=None,
                          names=['named-entity', 'art_ID'])
        coarseNE = spread_artID(coarseNE)
        # Remove the rows with NaN that contained the IDs of the articles initially.
        coarseNE = coarseNE.dropna()
        coarseNE['named-entity'] = coarseNE['named-entity'].map(lambda x: x.split('\t'))
        coarseNE['senIdx'] = coarseNE['named-entity'].map(lambda x: int(x[0]))
        coarseNE['begin'] = coarseNE['named-entity'].map(lambda x: int(x[1]))
        coarseNE['end'] = coarseNE['named-entity'].map(lambda x: int(x[2]))
        coarseNE['entityType'] = coarseNE['named-entity'].map(lambda x: get_tag(int(x[3])))
        coarseNE = coarseNE.drop(columns='named-entity')
        
        numOfEntities += coarseNE.shape[0]
        if count % 10 == 0:
            print(count, end=' ')
        count += 1
        
        pd.to_pickle(coarseNE, df_path)    
    print('Cantidad de entidades: {}'.format(numOfEntities))

    
def main():
    start = time.time()
#     create_docs_df('./corpus_WiNER/Documents/', './corpus_WiNER/docs_df/')
#     create_coarseNE_df('./corpus_WiNER/CoarseNE/', './corpus_WiNER/coarseNE_df/')
    end = time.time()
    print('Demora: {}'.format(end - start))


if __name__ == '__main__':
    main()

