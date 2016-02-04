# coding: utf-8

import os
import re
import pickle
import json
import pandas as pd
import numpy as np
import scipy as sp
import gensim as gs



# These can be taken from command line args or a config file if need be
os.chdir('..')
workingDir = os.getcwd()
sourceTextDir = "source_text"
pickleDir = 'models'
configDir = 'src/config'
configFile = 'output_config.json'
outputDir = 'shiny/data'
utf8Dir = 'text_utf8'
outputDir = 'shiny/data'

if not(os.path.exists(outputDir)):
    os.mkdir(outputDir)



# # Global config variables

params = json.load(open(os.path.join(workingDir,configDir,configFile),'r'))

bigramFile = False
if "bigramModel" in params:
    bigramFile = params["bigramModel"]

trigramFile = False
if "trigramModel" in params:
    trigramFile = params["trigramModel"]

if "tfidfModel" in params:
    tfidfFile = params["tfidfModel"]
else: tfidfFile = "tfidfs_gensim"
    
if "LDAModel" in params:
    LDAFile = params["LDAModel"]
else: LDAFile = "LDA_50_topics_auto_alpha_auto_eta"
    
if "dictionary" in params:
    dictionaryFile = params["dictionary"]
else: dictionaryFile = "dictionary_gensim"
    
if "corpus" in params:
    corpusFile = params["corpus"]
else: corpusFile = "corpus_gensim.p"
    
if "numTopicTerms" in params:
    numTopicTerms = params["numTopicTerms"]
else: numTopicTerms = 100
    
if "numDocTerms" in params:
    numDocTerms = params["numDocTerms"]
else: numDocTerms = 30
    
if "topicThreshold" in params:
    topicThreshold = params["topicThreshold"]
else: topicThreshold = .01



# # Read in models and corpus:

with open(os.path.join(workingDir,pickleDir,corpusFile), 'rb') as readfile:
    corpus_gs = pickle.load(file=readfile)

id2word = gs.corpora.Dictionary.load(os.path.join(pickleDir,dictionaryFile))

tfidf = gs.models.TfidfModel.load(os.path.join(pickleDir,tfidfFile))
#tfidf = gs.models.TfidfModel(dictionary=id2word) if non-existent or inconsistent

LDA = gs.models.LdaModel.load(os.path.join(pickleDir,LDAFile))

if "bigramModel" in params:
    bigrams = gs.models.Phrases.load(os.path.join(workingDir,pickleDir,bigramFile))
   
if "trigramModel" in params:
    trigrams = gs.models.Phrases.load(os.path.join(workingDir,pickleDir,trigramFile))



# Sanity check: do word counts in the gensim tokens and unicode tokens match?  
# Do a manual text search of the text file to double check
#i = 119
#token = 'specifications'
#filename = corpus.index[i]
#print(filename)
#tokens = (corpus.tokens[corpus.index[i]])
#counts = [np.str.count(str(word), token) for word in tokens]
#print(sum(counts))
#
#wordid = id2word.token2id[token]
#print(wordid)
#print(dict(corpus_gs[filename])[wordid])


# Another sanity check: should get 'ea' and 'true' as terms with mid-range df but very high tf
#for i in range(0,20000):
#    if id2word.dfs[i] < 30 and trigrams.vocab[id2word[i]] > 10000:
#        print(i)
#        print(id2word[i])



# # Generate outputs

# ### Term idfs

# Dictionary of corpus-wide idf's (excepting the common/stopwords and rare terms which have been removed)
idfOutput = {}

for key in id2word.keys():
    idfOutput[id2word[key]] = tfidf.idfs[key]


json.dump(idfOutput,open(os.path.join(outputDir,'idf.json'),'w'))

# In R, the following code
# list = fromJSON('idf.json',simplifyDataFrame = T,simplifyVector = T)
# list <- as.environment(list)
# will yield a hashable 'dictionary' (environment) of the idf's



# ### Document top terms

docTermsOutput = {}

for key in corpus_gs.keys():
    
    df = pd.DataFrame(tfidf[corpus_gs[key]], columns = ['id','tfidf'])
    df['token'] = [id2word[i] for i in df['id']]
    #print(type(df))
    df = df.sort_values(by='tfidf',axis=0,ascending=False)
    df = df.drop('id',axis=1)
    df = df.drop(df.index[np.arange(numDocTerms,df.shape[0])],axis = 0)
    df = df.to_dict(orient = 'records')
    docTermsOutput[key] = df

json.dump(docTermsOutput, open(os.path.join(outputDir,'docTerms.json'),'w'))



# ### Topic top terms

topicTermsOutput = {}
#topicStats = pd.DataFrame(index = range(0,LDA.num_topics), columns = ['entropy'])

for i in range(0,LDA.num_topics):
    
    df = pd.DataFrame(LDA.show_topic(i,topn=numTopicTerms), columns = ['weight','token'])
    #topicStats.loc[i,'entropy'] = sp.stats.entropy(df['weight'])
    df = df.to_dict(orient = 'records')
    #df['id'] = [id2word.token2id[token] for token in df['token']]
    topicTermsOutput[i] = df

json.dump(topicTermsOutput, open(os.path.join(outputDir,'topicTerms.json'),'w'))
#topicStats.to_json(os.path.join(outputDir,'topicStats.json'),orient='records')



# ### Document top topics

# Generate sparse doc-topic matrix for R
docTopics = {}
for filename in corpus_gs.keys():
    df = pd.DataFrame(LDA.get_document_topics(corpus_gs[filename], minimum_probability=topicThreshold), columns=['topic','weight'])
    df = df.to_dict(orient ='records')
    docTopics[filename] = df

json.dump(docTopics, open(os.path.join(outputDir,'docTopics.json'),'w'))



# ### Document stats

docStats = pd.DataFrame(index = corpus_gs.keys(), columns = ['length','topicEntropy'])

docIndex = dict(zip(docStats.index, range(0,docStats.shape[0])))


# Generate dense doc-topic matrix for entropy and JS-divergence computations

docTopicMatrix = {}

for filename in corpus_gs.keys():
    docTopicMatrix[filename] = dict(LDA.get_document_topics(corpus_gs[filename], minimum_probability=0.0))

docTopicMatrix = pd.DataFrame(docTopicMatrix).T

for filename in docStats.index:
    docStats.loc[filename,'length'] = np.sum(list(zip(*corpus_gs[filename]))[1])
    docStats.loc[filename,'topicEntropy'] = sp.stats.entropy(docTopicMatrix.loc[filename,:])
    docStats.loc[filename,'name'] = filename

docStats.to_json(os.path.join(outputDir,'docStats.json'),orient='records')



# ### Document similarities

# Jensen-Shannon divergence- a principled measure of distance between discrete probability distributions

def JSdivergence(p1,p2):
    P1 = p1/np.sum(p1)
    P2 = p2/np.sum(p2)
    M = .5*(P1+P2)
    return .5*(sp.stats.entropy(P1,M) + sp.stats.entropy(P2,M))


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

docDist.to_csv(os.path.join(outputDir,'docDist.csv'))

