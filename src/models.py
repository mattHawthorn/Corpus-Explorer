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
import gensim as gs
import cchardet
import nltk


# For logging gensim progress to terminal:
#logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


##########################################################
# Directories

# These can be taken from command line args or a config file if need be
os.chdir('..')
workingDir = os.getcwd()
sourceTextDir = "source_text"
pickleDir = 'models'
utf8Dir = 'text_utf8'
#utf16Dir = 'text_utf16'
configDir = 'src/config'
configFile = 'model_config.json'
logDir = 'log'
stopwordsInputFile = 'custom_stopwords.txt'
stopwordsOutputFile = 'removed_stopwords.txt'
logFile = 'model_log.txt'

os.chdir(workingDir)
print("Working dir is set to " + workingDir)

# Logger to log to terminal and log file
#class Logger:
#    def __init__(self,logPath):
#        self.terminal = sys.stdout
#        self.log = open(logPath, "a")
#
#    def write(self, message):
#        self.terminal.write(message)
#        self.log.write(message)  

#sys.stdout = Logger(logPath = os.path.join(workingDir,logDir,logFile))
#sys.stdout = open(os.path.join(workingDir,logDir,logFile),'w')


##########################################################
# Create directories if needed
if(not(os.path.exists(os.path.join(workingDir,pickleDir)))):
    os.mkdir(pickleDir)
if(not(os.path.exists(os.path.join(workingDir,utf8Dir)))):
    os.mkdir(utf8Dir)
if(not(os.path.exists(os.path.join(workingDir,configDir)))):
    os.mkdir(configDir)
if(not(os.path.exists(os.path.join(workingDir,logDir)))):
    os.mkdir(logDir)

if(not(os.path.exists(os.path.join(workingDir,configDir,configFile)))):
    print(os.path.join(configDir,configFile) + ' does not exist; aborting.')
    exit()


##########################################################
# Global config variables
params = json.load(open(os.path.join(workingDir,configDir,configFile),'r'))

# bigram and trigram detection thresholds; higher means more stringent requirement for detection
bigramThreshold = None
if 'bigramThreshold' in params:
    bigramThreshold = params['bigramThreshold']
trigramThreshold = None
if 'bigramThreshold' in params:
    trigramThreshold = params['trigramThreshold']

# remove docs from the corpus if they have fewer than this many tokens
minWordsPerDoc = params['minWordsPerDoc']

# skip tokens with fewer/more than this many characters
minWordLength = params['minWordLength']
maxWordLength = params['maxWordLength']

# df upper bound as a proportion of the corpus size:
dfThresholdHigh = params['dfThresholdHigh']
# df lower bound as a proportion of the corpus size, first pass (not necessarily removed):
dfThresholdLow = params['dfThresholdLow']
# tf lower bound for removal of terms with low df:
tfThreshold = params['tfThreshold']

# load custom stopwords?
specifyStopwords = params['specifyStopwords']
# save new stopwords?
updateStopwords = params['updateStopwords']

# generate LDA models with these parameters:
LDAParams = params['LDAParams']


##########################################################
# # Functions for text ingestion and cleanup

# Function: for each folder, concatenate contents
# ext is an extension 
def stdEncodeConcat(sourceDir,destinationDir,suffix='',ext='txt',sourceEncoding='utf-16',destinationEncoding='utf-8', temp = False, 
              detectEncoding=True, overwrite=False, verbose = True):
    # all files in source dir
    filenames = [f for f in os.listdir(sourceDir) if (f.endswith(ext))]
    
    # make destination dir if necessary
    if not(os.path.exists(destinationDir)):
        os.mkdir(destinationDir)
        
    # throw out the temp files if temp = False
    if not(temp):
        filenames = [f for f in filenames if not(f.startswith(".") | f.startswith("~"))]
        
    # strip off the last subfolder from the source folder as the destination file prefix 
    destFile = sourceDir.split('/')
    destFile = destFile[len(destFile) - 1]
    destFile = destFile + suffix + '.' + ext
    destPath = os.path.join(destinationDir,destFile)
    
    if not(overwrite):
        if os.path.exists(destPath):
            if(verbose):
                print(destPath + " already exists; skipping.")
            return
    
    with open(destPath, mode='w') as outfile:
        if(verbose):
            print("Concatenating " + str(len(filenames)) + ' ' + ext + " files in " + sourceDir)

        for filename in filenames:
            try:
                with open(sourceDir + '/' + filename, mode = "r") as infile:
                    string = infile.read()
                    if detectEncoding==True:
                        sourceEncoding = cchardet.detect(string)['encoding']
                        #if(verbose):
                        #    print(sourceEncoding)
                    string = string.decode(sourceEncoding)                   
                    outfile.write(string.encode(destinationEncoding))
            except DecodeError:
                continue


def stdEncode(sourceFile,destinationDir,suffix='',ext='txt',sourceEncoding='utf-16',destinationEncoding='utf-8', 
              detectEncoding=True, overwrite=False, verbose = True):
    print(destinationDir)
    print(sourceFile)
    # make destination dir if necessary
    if not(os.path.exists(destinationDir)):
        os.mkdir(destinationDir)
    
    destFile = re.sub('(\.[a-zA-Z0-9]{1,6})', suffix+'.'+ext, sourceFile)
    destPath = os.path.join(destinationDir,destFile)
    
    if not(overwrite):
        if os.path.exists(destPath):
            if(verbose):
                print(destPath + " already exists; skipping.")
            return
    
    with open(sourceFile, mode = "r") as infile:
        try:
            string = infile.read()
            if detectEncoding==True:
                sourceEncoding = cchardet.detect(string)['encoding']
                #print(sourceEncoding)
            
            string = string.decode(sourceEncoding)
            
            with open(destPath,"w") as outfile:
                outfile.write(string.encode('utf-8'))
        except DecodeError:
            return


# regexes for preprocessing
bracketed = re.compile(u'\[[^\[\]]*\]')
numeric = re.compile(u'[0-9]+')
apostrophe = re.compile(u'(\w)[\'](\w)')
abbreviations = re.compile(u'(\w)\.(\w)\.?')
punct = re.compile(u'[-﻿￼\.\?\!\,\"\ˈ\':ː\;\\\/\#\`\~\@\$\%\^\&\*\(\)_\+=\{\}\|\[\]\<\>\u2000-\u206F\u2E00-\u2E7F]')
whitespace = re.compile(u'\s+')

def clean_text(string):
    string = re.sub(bracketed, ' ', string)
    string = re.sub(numeric, ' ', string)
    string = re.sub(apostrophe, '\\1\\2', string)
    string = re.sub(abbreviations, '\\1\\2', string)
    string = re.sub(punct, ' ', string)
    string = re.sub(whitespace, ' ', string)
    string = string.lower().strip()
    return string




##########################################################
# # Concatenate text in subdirectories and standardize the encodings (if this hasn't already been done)

print("\nRetrieving files and folders in " + os.path.join(workingDir,sourceTextDir) + ":")
# get names of the individual text files in sourceTextDir
files = [f for f in os.listdir('./' + sourceTextDir)]
files = [f for f in files if (not(f.startswith(".")) and f.endswith(".txt"))]
print(str(len(files)) + " text files")

# get names of folders
folders = next(os.walk(os.path.join(workingDir,sourceTextDir)))[1]
print(str(len(folders)) + " subdirectories")



# write utf8-encoded concatenated versions

destDir = utf8Dir

print("\nConcatenating text in subdirectories to " + os.path.join(workingDir,destDir) + ":")
# concatenate texts in the subdirectories
for folder in folders:
    sourceDir = os.path.join(sourceTextDir,folder)
    stdEncodeConcat(sourceDir = sourceDir, destinationDir = destDir,destinationEncoding = 'utf-8', suffix = "(full)",ext = "txt")


# change encodings of the other single files to utf8

print("\nStandardizing encodings of remaining text files to " + os.path.join(workingDir,utf8Dir) + ":")
for filename in files:
    with open(os.path.join(destDir,filename), 'w') as outfile, open(os.path.join(sourceTextDir,filename),'r') as infile:
        outfile.write(infile.read())




##########################################################
# # Preprocess texts- removal of punctuation, numerals, etc.  Detection and joining of bigrams

textDir = utf8Dir
textFiles = os.listdir(textDir)


# set up dataframe to hold corpus (this one is small enough to put in memory for now)
corpus = pd.DataFrame(index = np.array(textFiles), columns = ['text','tokens'])


# for every doc in the text directory, tokenize it in lowercase without puntuation/numerals,
# and store the tokens as a list in the corresponding row of the corpus dataframe.
# Then compute IDF scores and remove terms with low scores as well as standard stop words if they remain.

print("\nBuilding corpus:")
for filename in corpus.index:
    with open(os.path.join(textDir, filename)) as document:
        text = document.read()
        print(type(text))
        #.decode('utf-8')
        tokens = gs.utils.simple_preprocess(clean_text(text), min_len=minWordLength, max_len=maxWordLength)
        #corpus.text[filename] = text
        corpus.tokens[filename] = tokens
        
print(str(corpus.shape[0]) + " documents in initial corpus.")



# Detect bigrams
if bigramThreshold:
    print("\nDetecting and joining bigrams, threshold = " + str(bigramThreshold))
    bigrams = gs.models.phrases.Phrases(sentences=corpus.tokens,threshold=bigramThreshold)

    for filename in corpus.index:
        corpus.loc[filename,'tokens'] = bigrams[corpus.tokens[filename]]


# Second pass can catch trigrams such as 'aberdeen proving ground' (and even possibly some quadrigrams by joining bigrams)
if trigramThreshold:
    print("Detecting and joining trigrams, threshold = " + str(trigramThreshold))
    trigrams = gs.models.phrases.Phrases(sentences=corpus.tokens,threshold=70.0)

    for filename in corpus.index:
        corpus.loc[filename,'tokens'] = trigrams[corpus.tokens[filename]]



##########################################################
# some documents may end up empty due to OCR or other issues; remove these

print("\nRemoving sparse documents with less than " + str(minWordsPerDoc) + " tokens:")
for filename in corpus.index:
    if len(corpus.tokens[filename]) < minWordsPerDoc:
        print(filename + "; only " + str(len(corpus.tokens[filename])) + " tokens.")
        corpus = corpus.drop(filename, axis = 0)

print("\n" + str(corpus.shape[0]) + " documents " + "remaining in the corpus.")



##########################################################
# Get dictionary and df stats on rare and common terms

# Generate corpus dictionary: maps ints to terms
print("\nGenerating corpus-wide dictionary:")
id2word = gs.corpora.Dictionary(corpus.tokens.values)
print(str(len(id2word)) + " initial terms in the dictionary.")


# sets to store potentially document-level counts of rare and frequent terms for removal
frequentWordKeys = set()
rareWordKeys = set()

# document frequency thresholds
dfMax = float(dfThresholdHigh)*float(id2word.num_docs)
dfMin = float(dfThresholdLow)*float(id2word.num_docs)

# Get excessively rare and excessively common terms and put in dicts with initial zero counts:
for key in id2word.keys():
    # These are candidates to add to the stopword list
    if id2word.dfs[key] > dfMax:
        frequentWordKeys.add(key)
    if id2word.dfs[key] < dfMin:
        rareWordKeys.add(key)
        
print(str(len(frequentWordKeys)) + " terms occur in more than " + str(100*dfThresholdHigh) + '% or ' + str(dfMax) + " documents.")
print(str(len(rareWordKeys)) + " terms occur in less than " + str(100*dfThresholdLow) + '% or ' + str(dfMin) + " documents.")


# frequentWordKeys seems to capture all the things we would like to remove: standard stopwords plus 'contractor',
# 'shall', 'contract', 'management', 'government', etc.
# Lots of typos and run-ons in the rareWordKeys list, plus a few non-ascii  characters




##########################################################
# import a standard stopwords list and join the very common terms to the stopwords list

# get initial English stopwords list from nltk
stopwords = set(nltk.corpus.stopwords.words('english'))

stopwordsPath = os.path.join(workingDir,configDir,stopwordsInputFile)

if specifyStopwords:
    print('\nLoading custom stopwords from ' + stopwordsPath)
    if os.path.exists(stopwordsPath):
        with open(stopwordsPath,'r') as infile:
            stopwords.update([(word.strip()) for word in infile.readlines()])

print("\nJoining common terms to standard stop words list.")
stopwordkeys = set([id2word.token2id[stopword] for stopword in stopwords if stopword in id2word.token2id.keys()])
stopwordkeys = stopwordkeys.union(frequentWordKeys)


# Save stopwords list
if updateStopwords:
    stopwordsPath = os.path.join(workingDir,logDir,stopwordsOutputFile)
    print("Saving removed stopwords list to " + stopwordsPath)
   
    with open(stopwordsPath,'w') as outfile:
        stopwords = sorted([id2word[stopwordkey] for stopwordkey in stopwordkeys])
        for stopword in sorted(stopwords):
            outfile.write(stopword + '\n')




##########################################################
# filter rare and common terms from dict

# get token id's for words occurring tfThreshold or less times maximum in any doc

temp = set()

if trigramThreshold:
    print('\nIdentifying sparse terms:')

    for wordID in rareWordKeys:
        #print(id2word[wordID] + ': ' + str(trigrams.vocab[id2word[wordID]]))
        if trigrams.vocab[id2word[wordID]] < tfThreshold:
            temp.add(wordID)
    
elif bigramThreshold:
    print('\nIdentifying sparse terms:')

    for wordID in rareWordKeys:
        #print(id2word[wordID] + ': ' + str(trigrams.vocab[id2word[wordID]]))
        if bigrams.vocab[id2word[wordID]] < tfThreshold:
            temp.add(wordID)

else:
    print(str(len(rareWordKeys)) + " terms occur in " + str(dfMin) + " or fewer documents; removing.")
   
if trigramThreshold or bigramThreshold:
    rareWordKeys = temp
    print(str(len(rareWordKeys)) + " of the rare terms occur " + str(tfThreshold) + " or fewer times in the corpus; removing.")

# Take the stopwords and rare words out of the dictionary
bad_ids = stopwordkeys.union(rareWordKeys)

print('\nRemoving stop words and sparse terms:')
id2word.filter_tokens(bad_ids=bad_ids)

# This will reassign token IDs to fill in the gaps left by the filtering:
id2word.compactify()
print(str(len(id2word)) + " terms remain in the dictionary after stopword and rare word removal.")

# compute a tfidf model with the new word IDs
print("\nComputing tf-idf model for terms in final dictionary")
tfidf = gs.models.tfidfmodel.TfidfModel(dictionary=id2word, normalize=False)




##########################################################
# Fit LDA models

# recompute bag-of-words for each doc with new dictionary (common and sparse terms removed)
corpus_gs = [id2word.doc2bow(corpus.tokens[filename]) for filename in corpus.index]

# # Run LDA models with gensim

models = {}

print("\nFitting LDA models using Gensim:")
for modelParams in LDAParams:
    alpha = modelParams['alpha']
    eta = modelParams['eta']
    num_topics = modelParams['topics']
    passes = modelParams['passes']
    key = 'LDA_' + str(num_topics) + '_topics_' + str(alpha) + '_alpha_' + str(eta) + '_eta_' + str(passes) + '_passes'

    if eta == 'default':
        eta = 100.0/float(len(id2word))
    
    if os.path.exists(os.path.join(pickleDir,key)):
        print(key + ' is already saved; skipping.')
        continue
    print("fitting model " + key)
    models[key] = gs.models.ldamodel.LdaModel(corpus_gs, num_topics=num_topics, id2word=id2word, passes=passes, alpha=alpha, eta = eta)




##########################################################
# # Save corpus and models

print("\nSaving text corpus.")
# convert the corpus to a dict indexed by text filenames
corpus_gs = dict(zip(corpus.index, corpus_gs))

# If you want to save actual tokens (large):
#with open(os.path.join(pickleDir,'corpus.p'), 'w') as savefile:
#    pickle.dump(corpus,file=savefile)
print("Saving Gensim (integer-keyed bag-of-words) corpus.")
with open(os.path.join(pickleDir,'corpus_gensim.p'), 'wb') as savefile:
    pickle.dump(corpus_gs,file=savefile)




print("Saving Gensim (integer-keyed) dictionary.")
id2word.save(os.path.join(pickleDir,'dictionary_gensim'))

if bigramThreshold:
    print("Saving Gensim bigrams detector.")
    bigrams.save(os.path.join(pickleDir,'bigrams_gensim'))
if trigramThreshold:
    print("Saving Gensim trigrams detector.")
    trigrams.save(os.path.join(pickleDir,'trigrams_gensim'))

print("Saving Gensim tf-idf model.")
tfidf.save(os.path.join(pickleDir,'tfidf_gensim'))

print("Saving Gensim LDA model objects:")
for key in models.keys():
    modelPath = os.path.join(pickleDir,key)
    print(modelPath)
    models[key].save(modelPath)



print("Done; exiting.")
exit()

