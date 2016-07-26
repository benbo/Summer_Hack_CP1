import os
from featurize_clusters import TextFeaturizer
from load_data import _extract_data_CP1
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import classification_report,f1_score,fbeta_score      
from collections import defaultdict
from operator import itemgetter
from itertools import groupby
import progressbar


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-ht',required=True)
    parser.add_argument('-lattice','-l',required=True)
    #parser.add_argument('-outfile','-o',required=True)            
    parser.add_argument('--text_features', '-t',default='Count')
    args = parser.parse_args()

    

    #load data
    # function filters and only loads certain keys to keep memory footprint minimal
    print 'loading HT data'
    if os.path.isdir(args.ht):
        filelist = [os.path.join(args.ht,f) for f in os.listdir(args.ht) if os.path.isfile(f)]
        #advertisements = {d[u'doc_id']:d for d in _extract_data_CP1(filelist)}
        advertisements = [d for d in _extract_data_CP1(filelist)]
    elif os.path.isfile(args.ht):
        #advertisements = {d[u'doc_id']:d for d in _extract_data_CP1([args.ht])}
        advertisements = [d for d in _extract_data_CP1([args.ht])]
    else:
        raise OSError(2, 'No such file or directory', args.ht)

    print 'loading lattice data'
    #load lattice extractions
    lattice = {d[u'_id']:d for d in load_gzip_json([args.lattice])}
    
    #get cluster ids and build a dictionary
    #this is going to be very slow
    #cluster_to_id = defaultdict(list)
    #for key,d in advertisements.iteritems():
    #    cluster_to_id[d[u'cluster_id']].append(key)
    
    #sort by cluster id
    print 'sorting ads'
    keyfunc = itemgetter(u'cluster_id')
    advertisements.sort(key=keyfunc)
    
    #iterate over clusters and create static features
    bar = progressbar.ProgressBar(redirect_stdout=True)
    print 'iterating over groups'
    for k, g in bar(groupby(advertisements, keyfunc)):
        #all features that only need to be generated once will be generated here
        group = tuple(g)
        keys = tuple(d[u'doc_id'] for d in group)
        g_lattice = tuple(lattice[k] for k in keys if k in lattice)#this may fail if not all lattice data is available 
        print 'group size: {}\nlattice group size: {}\n\n'.format(len(group),len(g_lattice))

        

    #TODO featurize clusters
    #
    
    
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
