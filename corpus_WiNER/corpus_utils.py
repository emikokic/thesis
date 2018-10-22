import pandas as pd
import os
import time
import numpy as np
from tqdm import tqdm
import gensim
import random
import itertools
from gensim.models import Word2Vec, KeyedVectors

from embeddings import concat_vectors, mean_vectors, fractional_decay, exponential_decay


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
    return ['PER', 'LOC', 'ORG', 'MISC'][x]


def tagToInt(tag):
    return {'PER': 0, 'LOC': 1, 'ORG': 2, 'MISC': 3, 'O': 4}[tag]    

    
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


def entityListFromSentence(senIdx, sen_length, art_entities_df):
    # We take the df with the entities of each sentence
    sen_entities_df = art_entities_df[art_entities_df.senIdx == senIdx]
    # An empty dataframe means that the sentence doesn't have any entity
    if sen_entities_df.empty:
        entities = ['O' for _ in range(sen_length)]
    else:
        entities = []
        i = 0
        for _, row in sen_entities_df.iterrows():
            while i < row['begin']:
                entities.append('O')
                i += 1
            while i < row['end']:
                entities.append(row['entityType'])
                i += 1
        while i < sen_length:
            entities.append('O')
            i += 1
    return entities


def w2v_strategy(strategy, sentence, W, w2v_model):
    return {
        'concat': concat_vectors(sentence, W, w2v_model), 
        'mean': mean_vectors(sentence, W, w2v_model), 
        'frac_decay': fractional_decay(sentence, W, w2v_model),
        'exp_decay': exponential_decay(sentence, W, w2v_model),
    }[strategy]


def getVector_EntityFromArticle(article_df, art_entities_df, strategy, W, w2v_model):
    '''@return: filled DataFrame with columns {wordVector, entityType}'''
    article_df = article_df.reset_index(drop=True)
    article_df['sen_length'] = article_df['sentence'].map(lambda x: len(x))

    fun = lambda sentence: w2v_strategy(strategy, sentence, W, w2v_model)
    article_df['sen_vectors'] = article_df['sentence'].map(fun)
    art_vectors = list(article_df['sen_vectors'])

    fun2 = lambda senIdx: entityListFromSentence(senIdx, article_df.loc[senIdx, 'sen_length'],
                                                 art_entities_df)
    article_df['sen_entities'] = article_df.index.map(fun2)
    art_entities = list(article_df['sen_entities'])
    # Fastest way to flatten list of arrays
    art_vectors = list(itertools.chain(*art_vectors))
    art_entities = list(itertools.chain(*art_entities))

    wordVector_Entity_df = pd.DataFrame({'wordVector': art_vectors, 
                                         'entityType': art_entities})

    return wordVector_Entity_df


def drop_non_entities(df, frac):
    '''
    Remove a fraction of non entities vectors (entityType == 'O')
    df: wordVector_Entity_df
    frac: float value between 0 and 1
    @return df with a fraction of the non entities rows removed
    '''
    sample = df[df.entityType == 'O'].sample(frac=frac, random_state=77)
    df = df.drop(index=sample.index)

    return df


def read_filenames():
    doc_filenames = os.listdir('./corpus_WiNER/docs_df/')
    doc_filenames = [int(f_name) for f_name in doc_filenames]
    doc_filenames.sort()
    doc_filenames = [str(f_name) for f_name in doc_filenames]
    coarseNE_filenames = os.listdir('./corpus_WiNER/coarseNE_df/')
    coarseNE_filenames = [int(f_name) for f_name in coarseNE_filenames]
    coarseNE_filenames.sort()
    coarseNE_filenames = [str(f_name) for f_name in coarseNE_filenames]
    
    return doc_filenames, coarseNE_filenames    


def genWordVectors_Entity(doc_df, coarseNE_df, strategy, W, w2v_model, splitType):
    '''
    Creates a N x D matrix of word vectors and saves it to disk. 
    Creates an 1 x N matrix of entities and saves it to disk. 
    N is the number of words in the document.
    D is the size of each word vector.
    The entity types are: PER - LOC - ORG - MISC - O
    
    strategy: 'concat', 'mean', 'frac_decay', 'exp_decay'.
    W: window size
    w2v_model: pre-trained word2vec model
    splitType: 'train', 'dev', 'test'
    '''
    wordVectors = []
    entityVector = []   
    art_IDs = coarseNE_df.art_ID.unique()      
    # We consider only the articles with at least one entity.
    # That's why we iterate over the coarseNE's articles.
    for art_ID in tqdm(np.nditer(art_IDs)):
        article_df = doc_df[doc_df.art_ID == art_ID]
        art_entities_df = coarseNE_df[coarseNE_df.art_ID == art_ID]     
        wordVector_Entity_df = getVector_EntityFromArticle(article_df, art_entities_df, 
                                                           strategy, W, w2v_model)         
        wordVector_Entity_df = drop_non_entities(wordVector_Entity_df, 0.80)
        wordVectors += list(wordVector_Entity_df['wordVector'])         
        entityVector += list(wordVector_Entity_df['entityType'])

    starting = time.time()
    np.savez_compressed('./corpus_WiNER/entity_vectors/ev_' + splitType + '_' + strategy 
                        + '_W_' + str(W), entityVector)
    np.savez_compressed('./corpus_WiNER/word_vectors/wv_'+ splitType + '_' + strategy
                        + '_W_' + str(W), wordVectors)
    finishing = time.time()
    print('tiempo de guardado:', finishing - starting)


def spread_words_entities(sentence, sen_entities, W):
    new_input = []
    L = len(sentence)
    # I: index of the target word
    for I in range(0, L):             
        words = []      
        # Padding with zeros on the left
        if I - W < 0:
            words += [''] * abs(I-W) #list(np.zeros(abs(I-W), dtype=int))
        # Concat vectors from the sentence
        for i in range(I - W, I + W + 1):             
            if i >= 0 and i < L:
                words.append(sentence[i])      
        # Padding with vector of zeros on the right
        if I + W >= L:
            words += list(np.zeros(abs(I+W+1-L), dtype=int))
        new_input.append((words, sen_entities[I]))
        
    return new_input


def generate_cnn_instances(art_IDs_sample, articles_df, entities_df, W, splitType):
    # NOTE: splitType parameter should be 'train' or 'dev' or 'test'
    # We consider only the articles with at least one entity.
    # That's why we iterate over the coarseNE's articles.
    sentences = []
    entities = []
    for art_ID in tqdm(np.nditer(art_IDs_sample)):
        article_df = articles_df[articles_df.art_ID == art_ID]
        art_entities_df = entities_df[entities_df.art_ID == art_ID] 
        article_df = article_df.reset_index(drop=True) # this is important for the entity matching.
        article_df['sen_length'] = article_df['sentence'].map(lambda x: len(x))
        fun = lambda senIdx: entityListFromSentence(senIdx, article_df.loc[senIdx, 'sen_length'],
                                                     art_entities_df)    
        article_df['entities'] = article_df.index.map(fun)
        sentences += list(article_df['sentence'])
        entities += list(article_df['entities'])

    instances = pd.DataFrame.from_dict({'sentence':sentences, 'entities':entities})
    new_input = []
    for idx, row in tqdm(instances.iterrows()):
        new_input += spread_words_entities(row['sentence'], row['entities'], W)

    instances = pd.DataFrame(new_input, columns=['words', 'entityType'])
    instances = drop_non_entities(instances, 0.80)
    instances.to_csv('./corpus_WiNER/cnn_instances/words_entity_W_'+str(W)+'_cnn_'+splitType+'.csv',
                     index=False)


def load_docs(doc_filenames, coarseNE_filenames):
    docs = []
    coarseNEs = []
    for doc, ne in zip(doc_filenames, coarseNE_filenames):
        docs.append(pd.read_pickle('./corpus_WiNER/docs_df/'+ doc))
        coarseNEs.append(pd.read_pickle('./corpus_WiNER/coarseNE_df/'+ ne))
    docs_df = pd.concat(docs, ignore_index=True)
    coarseNE_df = pd.concat(coarseNEs, ignore_index=True)
    
    return docs_df, coarseNE_df 


def main():
    start = time.time()
#     create_docs_df('./corpus_WiNER/Documents/', './corpus_WiNER/docs_df/')
#     create_coarseNE_df('./corpus_WiNER/CoarseNE/', './corpus_WiNER/coarseNE_df/')
    end = time.time()
    print('Demora: {}'.format(end - start))


if __name__ == '__main__':
    main()

