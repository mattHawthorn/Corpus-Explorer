# coding: utf-8

import os
import logging
import re
import sys
import pickle
import json
import operator
import logging
import pandas as pd
import numpy as np
import scipy.stats as stats
import gensim as gs
import cchardet



# Directories
workingDir = os.path.dirname(os.getcwd())
#sourceTextDir = "source_text"
pickleDir = 'models'
configDir = 'src/config'
outputConfigFile = 'output_config.json'
modelConfigFile = 'model_config.json'
outputDir = os.path.join(workingDir,'shiny/data/new')




# if no output dir, make one
if(not(os.path.exists(os.path.join(workingDir,outputDir)))):
    os.mkdir(outputDir)




# # Global config variables

# for models and output params:
params = json.load(open(os.path.join(workingDir,configDir,outputConfigFile),'r'))

bigramFile = False
if params.has_key("bigramModel"):
    bigramFile = params["bigramModel"]

trigramFile = False
if params.has_key("trigramModel"):
    trigramFile = params["trigramModel"]

if params.has_key("tfidfModel"):
    tfidfFile = params["tfidfModel"]
else: tfidfFile = "tfidfs_gensim"
    
if params.has_key("LDAModel"):
    LDAFile = params["LDAModel"]
else: LDAFile = "LDA_50_topics_auto_alpha_auto_eta"
    
if params.has_key("dictionary"):
    dictionaryFile = params["dictionary"]
else: dictionaryFile = "dictionary_gensim"
    
if params.has_key("corpus"):
    corpusFile = params["corpus"]
else: corpusFile = "corpus_gensim.p"
    
if params.has_key("numTopicTerms"):
    numTopicTerms = params["numTopicTerms"]
else: numTopicTerms = 100
    
if params.has_key("numDocTerms"):
    numDocTerms = params["numDocTerms"]
else: numDocTerms = 30
    
if params.has_key("topicThreshold"):
    topicThreshold = params["topicThreshold"]
else: topicThreshold = .01

# load models
id2word = gs.corpora.Dictionary.load(os.path.join(workingDir,pickleDir,dictionaryFile))

tfidf = gs.models.TfidfModel.load(os.path.join(workingDir,pickleDir,tfidfFile))
#tfidf = gs.models.TfidfModel(dictionary=id2word) if non-existent or inconsistent

LDA = gs.models.LdaModel.load(os.path.join(workingDir,pickleDir,LDAFile))

if params.has_key("bigramModel"):
    bigrams = gs.models.Phrases.load(os.path.join(workingDir,pickleDir,bigramFile))
   
if params.has_key("trigramModel"):
    trigrams = gs.models.Phrases.load(os.path.join(workingDir,pickleDir,trigramFile))


# for document ingestion:
params = json.load(open(os.path.join(workingDir,configDir,modelConfigFile),'r'))

# remove docs from the corpus if they have fewer than this many tokens
minWordsPerDoc = params['minWordsPerDoc']

# skip tokens with fewer/more than this many characters
minWordLength = params['minWordLength']
maxWordLength = params['maxWordLength']




# functions for text cleanup

# regexes for preprocessing:
bracketed = re.compile(u'\[[^\[\]]*\]')
numeric = re.compile(u'[0-9]+')
apostrophe = re.compile(u'(\w)[\'](\w)')
abbreviations = re.compile(u'(\w)\.(\w)\.?')
punct = re.compile(u'[-﻿￼\.\?\!\,\"\ˈ\':ː\;\\\/\#\`\~\@\$\%\^\&\*\(\)_\+=\{\}\|\[\]\<\>\u2000-\u206F\u2E00-\u2E7F]')
whitespace = re.compile(u'\s+')

# put them together in proper sequence for text cleaning:
def clean_text(string):
    string = re.sub(bracketed, ' ', string)
    string = re.sub(numeric, ' ', string)
    string = re.sub(apostrophe, '\\1\\2', string)
    string = re.sub(abbreviations, '\\1\\2', string)
    string = re.sub(punct, ' ', string)
    string = re.sub(whitespace, ' ', string)
    string = string.lower().strip()
    return string

# Jensen-Shannon divergence
def JSdivergence(p1,p2):
    P1 = p1/np.sum(p1)
    P2 = p2/np.sum(p2)
    M = .5*(P1+P2)
    return .5*(stats.entropy(P1,M) + stats.entropy(P2,M))



# The main function;
# For a list of paths to new docs, ingest the new docs, infer bigrams, trigrams, topics, top terms, and write to temp json's:

def ingestNewDocs(newFiles, newPaths):
    # build tokenized text corpus with inferred bigrams/trigrams
    newFiles = [newFile +'(NEW)' for newFile in newFiles]
    docIndex = dict(zip(newFiles,newPaths))
    corpus = pd.DataFrame(index = newFiles, columns = ['tokens'])
    
    for newFile in newFiles:
        newPath = docIndex[newFile]
        with open(newPath, 'r') as infile:
            string = infile.read()
            sourceEncoding = cchardet.detect(string)['encoding']
            string = string.decode(sourceEncoding)
            tokens = gs.utils.simple_preprocess(clean_text(string), min_len=minWordLength, max_len=maxWordLength)
            tokens = bigrams[tokens]
            tokens = trigrams[tokens]
            
            if len(tokens) >= minWordsPerDoc:
                corpus.tokens[newFile] = tokens
    
    corpus.dropna(axis='index',how='any',inplace=True)
    print(corpus)
    
    # build gensim corpus
    corpus_gs = [id2word.doc2bow(corpus.tokens[filename]) for filename in corpus.index]
    corpus_gs = dict(zip(corpus.index, corpus_gs))
    
    # Generate and write doc top terms to json
    docTermsOutput = {}
    
    for key in corpus_gs.keys():
        df = pd.DataFrame(tfidf[corpus_gs[key]], columns = ['id','tfidf'])
        df['token'] = [id2word[i] for i in df['id']]
        df = df.sort_values(by='tfidf',axis=0,ascending=False)
        df = df.drop('id',axis=1)
        df = df.drop(df.index[np.arange(numDocTerms,df.shape[0])],axis = 0)
        df = df.to_dict(orient = 'records')
        docTermsOutput[key] = df
    
    json.dump(docTermsOutput, open(os.path.join(workingDir,outputDir,'docTerms.json'),'w'))
    
    # Generate and write sparse doc-topic matrix to json
    docTopics = {}
    
    for filename in corpus_gs.keys():
        df = pd.DataFrame(LDA.get_document_topics(corpus_gs[filename], minimum_probability=topicThreshold), columns=['topic','weight'])
        df = df.to_dict(orient ='records')
        docTopics[filename] = df
    
    json.dump(docTopics, open(os.path.join(workingDir,outputDir,'docTopics.json'),'w'))
    
    # Generate document stats and write to json
    docStats = pd.DataFrame(index = corpus_gs.keys(), columns = ['length','topicEntropy'])
    docTopicMatrix = {}
    
    for filename in corpus_gs.keys():
        docTopicMatrix[filename] = dict(LDA.get_document_topics(corpus_gs[filename], minimum_probability=0.0))
    
    docTopicMatrix = pd.DataFrame(docTopicMatrix).T
    
    for filename in docStats.index:
        docStats.loc[filename,'length'] = np.sum(zip(*corpus_gs[filename])[1])
        docStats.loc[filename,'topicEntropy'] = stats.entropy(docTopicMatrix.loc[filename,:])
        docStats.loc[filename,'name'] = filename
    
    docStats.to_json(os.path.join(workingDir,outputDir,'docStats.json'),orient='records')
    
    n = len(corpus_gs.keys())
    # Prepare distance matrix
    docDist = np.zeros(shape = (n,n), dtype='float16')

    # get upper triangle indices
    upperTriIndices = np.triu_indices(n,k=1)
        
    # Compute JS divergences for all pairs, normalized to [0,1] interval
    for i in zip(upperTriIndices[0],upperTriIndices[1]):
        p1 = docTopicMatrix.iloc[i[0],].values
        p2 = docTopicMatrix.iloc[i[1],].values
        # normalize to 0-1 range by dividing by log(2)
        docDist[i[0],i[1]] = docDist[i[1],i[0]] = JSdivergence(p1,p2)/np.log(2)
    
    docDist = pd.DataFrame(docDist, index = docTopicMatrix.index, columns = docTopicMatrix.index)
    
    docDist.to_csv(os.path.join(workingDir,outputDir,'docDist.csv'))

#newPaths = ['/home/matthew/Desktop/test.txt','/home/matthew/Desktop/test2.txt']
#ingestNewDocs(newPaths, newPaths)
