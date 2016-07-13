import os
import json
from featurize_clusters import TextFeaturizer

path="C:/Users/maria/repos/Summer_Hack_CP1/"
os.chdir(path)


jsonfilename = 'test_json.txt'
with open(jsonfilename, 'r') as infile:
    data = [json.loads(line) for line in infile]

#take a look at the first one
#data[0]
#take a look at the keys
#data[0].keys()

###Extract all text as a list

##data_text = list()
##for i in xrange(0,len(data)):
##    data_text.append(data[i]["extracted_text"])

data_text = [d["extracted_text"] for d in data]





###Fit vectorizers to all training data
A =  TextFeaturizer(data_text)#Count Vectorizer
X = A.get_text_features(data_text) #usually data_text would be replaced by cluster

#for each cluster:
           #2. obtain matrix for cluster
           #3. summarize


##
##documents = ['I am a test string','I am another test string','and a third test string']
##    small_docs = ['I am a test string','I am another test string']
##    A =  TextFeaturizer(documents)#Count Vectorizer
##    X = A.get_text_features(small_docs)
##    B =  TextFeaturizer(documents,options=BASE_OPTIONS_TFIDF,Count=False)#TFIDF Vectorizer
##    X = B.get_text_features(small_docs)
