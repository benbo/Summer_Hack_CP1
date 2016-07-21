import os
import json
from featurize_clusters import TextFeaturizer
import numpy as np
from itertools import chain
from pyproj import Proj,transform


INPROJ = Proj(init='epsg:4326')
OUTPROJ = Proj(init='epsg:3857')

def cluster_text_summary(textfeaturizer,text):
    '''
    create vector embedding for all text in a cluster of ads    
    '''
    #Text features
    X = textfeaturizer.get_text_features(text) #usually data_text would be replaced by cluster

    X = X.tocsc() #transform into column-sparse object
    mean_text = X.mean(axis=0) #mean for cluster
    #sqaure elements to then compute variance
    X.data**=2
    variance_text = X.mean(axis=0)-np.multiply(mean_text,mean_text)  #variance of each word
    agg_variance = np.sum(variance_text) #aggregate variance

    #create vector of text features for the cluster
    return np.hstack((np.squeeze(np.array(mean_text)),agg_variance))

def cluster_age_summary(lattice_cluster):
    ages = [res for res in get_age(lattice_cluster)]
    missing_age = len(lattice_cluster)-len(ages) #number of ads that do not contain age
    ages = np.array(tuple(chain(*ages)))

    #vector that summarizes the age features of cluster
    return np.array((ages.mean(),ages.std(),ages.median(),missing_age))

def cluster_location_summary(lattice_cluster):
    locations_webMer = [coord for coord in get_coord(lattice_cluster)] 

    num_found_loc = len(locations_webMer)
    missing_lat = len(lattice_cluster)-num_found_loc
    missing_lat_proportion = float(missing_lat)/len(lattice_cluster)

    locations_webMer=np.array(locations_webMer)

    #Extract mean of xaxis
    centroid_cluster = np.mean(locations_webMer,0)

    ###Measure the area the cluster covers
    min_coord = np.amin(locations_webMer,0)
    max_coord = np.amax(locations_webMer,0)

    dif_extreme_coord = np.subtract(max_coord,min_coord)
    area_cluster = dif_extreme_coord[0]*dif_extreme_coord[1]

    ###Measure how concentrated ads are
    #distance from each point to the centroid - add 1 to avoid errors when all ads are in same location 

    sqdif_coord = np.zeros(shape=(num_found_loc,2))
    sqdif_coord[:,0] = np.square(np.abs(locations_webMer[:,0]-centroid_cluster[0]))
    sqdif_coord[:,1] = np.square(np.abs(locations_webMer[:,1]-centroid_cluster[1]))
    dist_centroid = np.sqrt(np.sum(sqdif_coord,1))
    
    loc_prob = [prob for prob in get_coord_prob(lattice_cluster)]#Extract probabilities of location
    
    disp_metric = sum(np.multiply(dist_centroid,loc_prob))/num_found_loc  ###alternatively we can sum over the log

    return np.hstack((missing_lat_proportion,np.squeeze(centroid_cluster),area_cluster,disp_metric))


def get_age(cluster):
    for item in cluster:
        if 'lattice-age' in item['extractions']:
            yield map(float,(d['value'] for d in item['extractions']['lattice-age']['results']))

def get_coord(cluster): 
        for item in cluster:
            if 'lattice-location' in item['extractions']:
                tup = tuple(transform(INPROJ,OUTPROJ,d['context']['city']['centroid_lon'],
                                                     d['context']['city']['centroid_lat']) 
                            for d in item['extractions']['lattice-location']['results']
                                if d['context']['city']['centroid_lat'] is not None)
                if tup:#add code to aggregate multiple locations into one
                    yield tup[0]

def get_coord_prob(cluster):
        for item in cluster:
            if 'lattice-location' in item['extractions']:
                prob = map(float,(d['probability'] for d in item['extractions']['lattice-location']['results'] if d['context']['city']['centroid_lat'] is not None))
                if prob:
                    yield prob[0] ###MODIFY to extract the prob needed, according to how locations are aggregated
