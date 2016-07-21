import os
import json
from featurize_clusters import TextFeaturizer
import numpy as np
from itertools import chain
from pyproj import Proj,transform


INPROJ = Proj(init='epsg:4326')
OUTPROJ = Proj(init='epsg:3857')
MV = 'None' # handle missing values this way

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

def temporal_summary(lattice_cluster):
	epoch_timestamps=list()
	posttime = gen_get( ['posttime'] , lattice_cluster )#Following is unique to posttime feature
	for item in posttime:
		time_elements= list()
		entities = item.split()#item.split() is of form ['Monday,','June','26,','2016','6:52','PM']
		try:
			hm_split = entities[4].split(':')
			time_elements.extend( [entities[2].strip(',')] )#day
			time_elements.extend( [month2int(entities[1])] )#month
			time_elements.extend( [entities[3]] )#year
			time_elements.extend( [hm_split[0]] )#hour
			time_elements.extend( [hm_split[1]] )#minute
			time_elements.extend( [0] ) #second...missing from tags but for completeness, set 0
			time_elements = map( int , time_elements )#Mix of strings/ints to all ints

			#Convert to military time
			if entities[5]=='PM' and time_elements[3]<12:
				time_elements[3] = time_elements[3]+12
			elif entities[5]=='AM' and time_elements[3]==12:
				time_elements[3] = 0

			#Convert to unix epoch time
			epoch_timestamps.extend( [make_timestamp(time_elements)] )
		except:
			epoch_timestamps.extend( [MV] )

	#String missing values and sort remaining
	sorted_epoch_posttimes = filter( lambda a: a!='None', epoch_timestamps )
	sorted_epoch_posttimes.sort()

	#1hour = 3600, 1day = 86400, 1week = 604800, 1month = 2629743, 1year = 31556926
	num_posts_in_24hr_windows = get_sliding_window_count( sorted_epoch_posttimes , 86400 )
	#for example, num_posts_in_1hr_window = get_sliding_window_count(sorted_epoch_posttimes,3600)

	max_posts_per_24hr_period = max( num_posts_in_24hr_windows )
	min_posts_per_24hr_period = min( num_posts_in_24hr_windows )
	avg_posts_per_24hr_period = np.mean( num_posts_in_24hr_windows )
	med_posts_per_24hr_period = np.median( num_posts_in_24hr_windows )
	std_posts_per_24hr_period = np.std( num_posts_in_24hr_windows )

	cluster_time_summary = [ max_posts_per_24hr_period ,
						min_posts_per_24hr_period ,
						avg_posts_per_24hr_period ,
						med_posts_per_24hr_period ,
						std_posts_per_24hr_period ]

	return cluster_time_summary



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




#Expand the json entries to a list of all entries of the same tag
def gen_get( keys , raw_data ):
	feature = list()
	for record in raw_data:
		if keys[0] in record['extractions']:
			if len(keys)==1:
				try:
					candidate = [str(record['extractions'][keys[0]]['results'][0])]
					if( candidate[0].find('html')<0 ): #Some fields contain html for the whole scraped page
						feature.extend(candidate)
				except:
					feature.extend([MV])
			elif len(keys)==2:
				try:
					candidate = [str(record['extractions'][keys[0]]['results'][0][keys[1]])]
					if( candidate[0].find('html')<0 ):
						feature.extend(candidate)
				except:
					feature.extend([MV])
			elif len(keys)==3:
				try:
					candidate = [str(record['extractions'][keys[0]]['results'][0][keys[1]][keys[2]])]
					if( candidate[0].find('html')<0 ):
						feature.extend(candidate)
				except:
					feature.extend([MV])
			elif len(keys)==4:
				try:
					candidate = [str(record['extractions'][keys[0]]['results'][0][keys[1]][keys[2]][keys[3]])]
					if( candidate[0].find('html')<0 ):
						feature.extend(candidate)
				except:
					feature.extend([MV])
			else:
				feature.extend([MV])
		else:
			feature.extend([MV])
	return feature


#Convert the month name to a month integer
def month2int( monthstr ):
	if monthstr == 'January':
		return 0 #zero indexed, since months represent offset from Jan. 1
	elif monthstr == 'February':
		return 1
	elif monthstr == 'March':
		return 2
	elif monthstr == 'April':
		return 3
	elif monthstr == 'May':
		return 4
	elif monthstr == 'June':
		return 5
	elif monthstr == 'July':
		return 6
	elif monthstr == 'August':
		return 7
	elif monthstr == 'September':
		return 8
	elif monthstr == 'October':
		return 9
	elif monthstr == 'November':
		return 10
	elif monthstr == 'December':
		return 11
	else:
		return 0 #If the month is missing for some reason, assume January


# Parse feature and create a numerical epoch timestamp
# Input looks like [ <day> , <month> , <year> , <hour> , <minute> , <second> ]
# All elements should be integer type
def make_timestamp( time_entity_list ):
	
	day = time_entity_list[0]
	month = time_entity_list[1]
	year = time_entity_list[2]
	hour = time_entity_list[3]
	minute = time_entity_list[4]
	second = time_entity_list[5]

	#Converting all units of time to seconds, i.e. there are 3600 seconds in an hour
	timestamp = hour*3600 + day*86400 + month*2629743 + year*31556926 + minute*60 + second
	return timestamp



# Find distances, d,  between each sequential pair of elements in a list, l
# i.e. gives you [ d(l[1],l[2]) , d(l[2],l[3]) , d(l[3],l[4]) , d(l[4],l[5]) , ... ]
def get_sequential_pairwise_distance( myList ):
	distances = list()
	for i in range(0,len(myList)-1):
		distances.extend([myList[i+1]-myList[i]])
	return distances


# Find sequential pairwise distances, then fill windows to see how many occur
# within a given period of time
def get_sliding_window_count( myList , windowSize ):
	element_distances = get_sequential_pairwise_distance( myList )
	window_count = list()
	for i in range(0,len(element_distances)-1):
		j=1
		span = element_distances[i]
		while span<windowSize and (i+j+1)<len(element_distances):
			j+=1
			span += element_distances[i+j]
		window_count.extend([j])
	return window_count



#***************************************************
# Uncomment for quick test of temporal summary
#**************************************************
#import random
#jsonfilename2 = '/zfsauton/project/public/memex/lattice/example_1000.json'
#with open(jsonfilename2, 'r') as infile:
	#data_lattice = [json.loads(line) for line in infile ]
#cluster = random.sample(data_lattice,1000)
#summary = temporal_summary( cluster )
#print 'max, min, avg, med, std'
#print summary
