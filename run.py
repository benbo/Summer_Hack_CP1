import os
from featurize_clusters import TextFeaturizer
from load_data import load_gzip_json
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score,fbeta_score      

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-labeled','-l',required=True)
    #parser.add_argument('-outfile','-o',required=True)            
    parser.add_argument('--text_features', '-t',default='Count')
    args = parser.parse_args()

    #add keys to be skipped to reduce memory footprint
    skip = frozenset([u'raw_content',u'content_type'])
    #load data
    if os.path.isdir(args.labeled):
        filelist = [os.path.join(args.labeled,f) for f in os.listdir(args.labeled) if os.path.isfile(f)]
        advertisements = {d[u'_id']:d for d in load_gzip_json(filelist,skip)}
    elif os.path.isfile(args.labeled):
        advertisements = {d[u'doc_id']:d for d in load_gzip_json([args.labeled],skip)}
    else:
        raise OSError(2, 'No such file or directory', args.labeled)
    
    #get cluster ids and build a dictionary
    #this is going to be very slow
    cluster_to_id = {}
    for key,d in advertisements.iteritems():
        c_key = d[u'cluster_id']
        if c_key in cluster_to_id:
            cluster_to_id[c_key].append(key)
        else:
            cluster_to_id[c_key] = [key]

    #TODO ER and fold generation
    #obtain clusters and generate folds
    #Cluster ids are available in the json objects

    #TODO join with lattice extractions
    

    #TODO featurize clusters
    #for fold in folds:
    folds=False
    if folds:
        #get training data
        
        #get text features
        #1. fit vectorizers to all training text
        #for each cluster:
           #2. obtain matrix for cluster
           #3. summarize
        
        #get test data
        #featurize test data

        #generate spatio temporal features

        #generate additional features

        #train classifier
        #l1
        #model = svm.LinearSVC(C=1.0, penalty='l1',dual=False,n_jobs=8)
        #l2
        #model = svm.LinearSVC(C=1.0, penalty='l2',n_jobs=8)

        #tune hyper parameter
        # run randomized search
        n_iter_search = 10
        param_grid = {'C':np.logspace(1.0, 500.0, num=n_iter_search)}
        grid_search = GridSearchCV(model, param_grid=param_grid)
        grid_search.fit(X_train, y_train)
        
        #predict test cases
        y_pred = grid_search.predict(X_test)
        
        print classification_report(y_test, y_pred)

        #get scores for evalution at end of our fold loop
        print f1_score(y_test,y_pred)

        #beta parameter determines the weight of precision in the combined score. 
        #beta < 1 lends more weight to precision, while beta > 1 favors recall (beta -> 0 considers only precision, beta -> inf only recall).
        beta = 0.5 
        print fbeta_score(y_test, y_pred, beta)

                

if __name__ == '__main__':
    main()
