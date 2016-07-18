import os
import json
from featurize_clusters import TextFeaturizer
import numpy as np
from itertools import chain
from pyproj import Proj,transform


path="/home/mdeartea/Documents/Memex/"
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

############operations per cluster
###add extraction of each cluster


#######Text features
X = A.get_text_features(data_text) #usually data_text would be replaced by cluster

X = X.tocsc() #transform into column-sparse object
mean_text = X.mean(axis=0) #mean for cluster
#sqaure elements to then compute variance
X.data**=2
variance_text = X.mean(axis=0)-np.multiply(mean_text,mean_text)  #variance of each word
agg_variance = np.sum(variance_text) #aggregate variance

#create vector of text features for the cluster
text_summary_cluster = np.hstack((np.squeeze(np.array(mean_text)),agg_variance))

#####
#Upload lattice extractions - in new version those will be used from the start
jsonfilename2 = 'example_1000.json'
with open(jsonfilename2, 'r') as infile:
    data_lattice = [json.loads(line) for line in infile]
    
#######Age
cluster = data_lattice[:100]  ####this would normally be within the iteration

def get_age(cluster):
    for item in cluster:
        if 'lattice-age' in item['extractions']:
            yield map(float,(d['value'] for d in item['extractions']['lattice-age']['results']))
        
ages = [res for res in get_age(cluster)]
missing_age = len(cluster)-len(ages) #number of ads that do not contain age
ages = tuple(chain(*ages))
mean_age = sum(ages)/len(ages)
ages_sq = tuple(a**2 for a in ages)
std_age = math.sqrt(sum(ages_sq)/len(ages)-mean_age**2)

###vector that summarizes the age features of cluster
age_summary_cluster = [mean_age,std_age,missing_age]

##############Location cluster features

def get_coord(axis,cluster): #axis is either centroid_lat or centroid_lon
    for item in cluster:
        if 'lattice-location' in item['extractions']:
            yield (d['context']['location'][axis] for d in item['extractions']['lattice-location']['results']  if d['context']['location'][axis] is not None)

def get_coord_prob(cluster): #axis is either centroid_lat or centroid_lon
    for item in cluster:
        if 'lattice-location' in item['extractions']:
            yield map(float,(d['probability'] for d in item['extractions']['lattice-location']['results'] if d['context']['location']['centroid_lat'] is not None))

###Extract latitudes for all ads
latitudes = [lat for lat in get_coord('centroid_lat',cluster)]
missing_lat = len(cluster)-len(latitudes)

latitudes = tuple(chain(*latitudes))  

###Extract longitude for all ads
longitudes = [lon for lon in get_coord('centroid_lon',cluster)]
missing_lon = len(cluster)-len(longitudes)

longitudes = tuple(chain(*longitudes)) 

###Calculate range of latitude (max - min)
range_lat = max(latitudes)-min(latitudes)

###Calculate range of longitude (max-min)
range_lon = max(longitudes)-min(longitudes)

###calculate area of rectangle
area_cluster = range_lat*range_lon

###Extract probabilities of location
loc_prob = [prob for prob in get_coord_prob(cluster)]
loc_prob = tuple(chain(*loc_prob)) 

###Calculate centroid of cluster - analogous to center of mass
loc_prob = np.array(loc_prob)
latitudes = np.array(latitudes)
longitudes = np.array(longitudes)

centroid_lat = sum(np.multiply(latitudes,loc_prob))/sum(loc_prob)
centroid_lon = sum(np.multiply(longitudes,loc_prob))/sum(loc_prob)

###Calculate how concentrated the cluster is

#calculate distance of each point to center of cluster
lat_diff = [(l - centroid_lat)**2 for l in latitudes]
lon_diff = [(l - centroid_lon)**2 for l in longitudes]

sq_dist_centroid = np.add(lat_diff,lon_diff)
dist_centroid = [math.sqrt(sq_dist) for sq_dist in sq_dist_centroid]

disp_metric = sum(np.multiply(dist_centroid,loc_prob))  
#disp_metric = sum(np.multiply(np.log(dist_centroid),loc_prob))  ###alternatively we can sum over the log

###########################################
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
