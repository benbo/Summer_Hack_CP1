from featurize_clusters import TextFeaturizer
from load_data import load_gzip_json
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-infile','-i',required=True)                                                                                                                                                       
    parser.add_argument('-outfile','-o',required=True)                                                                                                                                                      
    parser.add_argument('--text_features', '-t',default='Count')
    args = parser.parse_args()

    #load data
    #TODO only load fields that are necessary to keep memory footprint low
    data = load_gzip_json(args.infile)

    #TODO ER and fold generation
    #obtain clusters and generate folds

    #TODO join with lattice extractinos
    

    #TODO featurize clusters
    #for fold in folds:
        #get training data
        
        #get text features
        #1. fit vectorizers to all training text
        #for each cluster:
           #2. obtain matrix for cluster
           #3. summarize

        #generate spatio temporal features

        #generate additional features

        #train classifier

        #test classifier


if __name__ == '__main__':
    main()
