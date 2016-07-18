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


########LOCATION features


inProj = Proj(init='epsg:4326')
outProj = Proj(init='epsg:3857')

def get_coord(cluster): #axis is either centroid_lat or centroid_lon
    for item in cluster:
        if 'lattice-location' in item['extractions']:
            tup = tuple(transform(inProj,outProj,d['context']['city']['centroid_lon'],
                             d['context']['city']['centroid_lat']) for d in item['extractions']['lattice-location']['results']  if d['context']['city']['centroid_lat'] is not None)
            if tup:#add code to aggregate multiple locations into one
                yield tup[0]
locations_webMer = [coord for coord in get_coord(cluster)] 

num_found_loc = len(locations_webMer)
missing_lat = len(cluster)-num_found_loc
missing_lat_proportion = np.float(missing_lat)/len(cluster)

def get_coord_prob(cluster): #axis is either centroid_lat or centroid_lon
    for item in cluster:
        if 'lattice-location' in item['extractions']:
            prob = map(float,(d['probability'] for d in item['extractions']['lattice-location']['results'] if d['context']['city']['centroid_lat'] is not None))
            if prob:
                yield prob[0] ###MODIFY to extract the prob needed, according to how locations are aggregated
###Extract probabilities of location
loc_prob = [prob for prob in get_coord_prob(cluster)]

locations_webMer=np.array(locations_webMer)

#Extract mean of xaxis
centroid_cluster = np.mean(locations_webMer,0)

###Measure the area the cluster covers
min_coord = np.amin(locations_webMer,0)
max_coord = np.amax(locations_webMer,0)

dif_extreme_coord = np.subtract(max_coord,min_coord)
area_centroid = dif_extreme_coord[0]*dif_extreme_coord[1]

###Measure how concentrated ads are
#distance from each point to the centroid - add 1 to avoid errors when all ads are in same location 

sqdif_coord = a = np.zeros(shape=(num_found_loc,2))
sqdif_coord[:,0] = [(np.abs(x - centroid_cluster[0])+1)**2 for x in locations_webMer[:,0]]
sqdif_coord[:,1] = [(np.abs(x - centroid_cluster[1])+1)**2 for x in locations_webMer[:,1]]

dist_centroid = [np.sqrt(sq_dist) for sq_dist in sq_dist_centroid]
disp_metric = sum(np.multiply(dist_centroid,loc_prob))/num_found_loc  ###alternatively we can sum over the log



location_summary_cluster = np.hstack((missing_lat_proportion,np.squeeze(centroid_cluster),area_cluster,disp_metric))





