import os
import json
from featurize_clusters import TextFeaturizer
import numpy as np
from itertools import chain
from pyproj import Proj,transform
import time
from datetime import datetime 
from pandas import Series, DataFrame, Panel
import pandas as pd

INPROJ = Proj(init='epsg:4326')
OUTPROJ = Proj(init='epsg:3857')
MV = 'None' # handle missing values this way

def get_static_features(cluster):
    return np.hstack((cluster_age_summary(cluster),cluster_location_summary(cluster),temporal_summary(cluster)))

def temporal_summary(lattice_cluster):
    #use strings of dates as they can be sorted without conversion
    posttimes = [d[u'extractions'][u'lattice-postdatetime'][u'results'][0][u'value'] for d in lattice_cluster if u'lattice-postdatetime' in d[u'extractions']]
    if len(posttimes)>1:
        dind = pd.to_datetime(sorted(posttimes))
        daterange = (dind[-1]-dind[0]).days#difference between first and last post in days
        df = Series(np.ones(len(posttimes)), index=dind)

        #1day -aggregate by day
        df1 = df.resample('d').sum().fillna(0.0)
        #2days -aggregate by 2-day window
        df2 = df1.resample("2d",closed='left').sum()
        #monthly -aggregate by month
        dfm = df1.resample("M",closed='left').sum() 

        return np.array((0.0,daterange,df1.mean(),df1.min(),df1.max(),df1.median(),df1.std(),df2.mean(),df2.min(),df2.max(),df2.median(),df2.std(),dfm.mean(),dfm.min(),dfm.max(),dfm.median(),dfm.std()))
    else:
        return np.vstack((np.array([1.0]),np.zeros(16)))


def cluster_text_summary(textfeaturizer,text):
    '''
    create vector embedding for all text in a cluster of ads    
    '''
    #Text features
    X = textfeaturizer.get_text_features(text)

    X = X.tocsc() #transform into column-sparse object
    mean_text = X.mean(axis=0) #mean for cluster
    #sqaure elements to then compute variance
    X.data**=2
    variance_text = X.mean(axis=0)-np.multiply(mean_text,mean_text)  #variance of each word
    agg_variance = np.sum(variance_text) #aggregate variance

    #create vector of text features for the cluster
    return np.hstack((np.squeeze(np.array(mean_text)),agg_variance))

def cluster_age_summary(lattice_cluster):
    ages = tuple(res for res in get_age(lattice_cluster))
    missing_age = len(lattice_cluster)-len(ages) #number of ads that do not contain age
    ages = np.array(tuple(chain(*ages)))

    #vector that summarizes the age features of cluster
    return np.array((ages.mean(),ages.std(),np.median(ages),missing_age))

def cluster_location_summary(lattice_cluster):
    locations_webMer,loc_prob = zip(*tuple((coord,prob) for coord,prob in get_locations(lattice_cluster)))

    num_found_loc = len(locations_webMer)
    missing_lat = len(lattice_cluster)-num_found_loc
    missing_lat_proportion = float(missing_lat)/len(lattice_cluster)

    locations_webMer=np.array(locations_webMer)

    #Extract mean of xaxis
    #centroid_cluster = np.mean(locations_webMer,0)
    centroid_cluster = np.average(locations_webMer,0,loc_prob)
    
    
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
    
    disp_metric = np.sum(np.multiply(dist_centroid,loc_prob))/num_found_loc  ###alternatively we can sum over the log

    #return np.hstack((missing_lat_proportion,np.squeeze(centroid_cluster),area_cluster,disp_metric))
    return np.hstack((missing_lat_proportion,area_cluster,disp_metric))


def get_age(cluster):
    for item in cluster:
        if 'lattice-age' in item[u'extractions']:
            yield map(float,(d[u'value'] for d in item[u'extractions'][u'lattice-age'][u'results']))

def get_coord(cluster): 
        for item in cluster:
            if u'lattice-location' in item[u'extractions']:
                
                tup = tuple(transform(INPROJ,OUTPROJ,d[u'context'][u'city'][u'centroid_lon'],
                                                     d[u'context'][u'city'][u'centroid_lat']) 
                            for d in item[u'extractions'][u'lattice-location'][u'results']
                                if d[u'context'][u'city'][u'centroid_lat'] is not None)
                if tup:#add code to aggregate multiple locations into one
                    yield tup[0]


def get_locations(cluster): 
    for item in cluster:
        if u'lattice-location' in item[u'extractions']:
            ret = tuple((coord,prob) for coord,prob in load_cities(item[u'extractions'][u'lattice-location'][u'results']))
            if ret:
                coords,probs = zip(*ret)
                coords = np.array(coords)
                probs = np.array(probs)
                yield np.average(coords,0,probs),probs.mean()

def load_cities(iterable):
    for d in iterable:
        if u'city' in d[u'context']:
            yield transform(INPROJ,OUTPROJ,d[u'context'][u'city'][u'centroid_lon'],
                            d[u'context'][u'city'][u'centroid_lat']),d[u'probability']

