from sklearn.feature_extraction.text import CountVectorizer
BASE_OPTIONS_COUNT = {                        
            'strip_accents': None,                 
            'stop_words': 'english',               
            'ngram_range': (1, 1),                 
            'analyzer': 'word',                    
            'max_df': 1.0,                         
            'min_df': 1,                           
            'max_features': None,                  
            'binary': False                        
            }

BASE_OPTIONS_TFIDF = {                             
            'strip_accents': None,                 
            'stop_words': 'english',               
            'ngram_range': (1, 1),                 
            'analyzer': 'word',                    
            'max_df': 1.0,                         
            'min_df': 1,                           
            'max_features': None,                  
            'binary': False, 
            'norm' : 'l2',
            'sublinear_tf' : False
            }  

class TextFeaturizer(object):
    def __init__(self,documents,options=BASE_OPTIONS_COUNT,Count=True):
        if Count:
            self.vectorizer = CountVectorizer(strip_accents=options['strip_accents'], stop_words=options['stop_words'], 
                              ngram_range=options['ngram_range'], analyzer=options['analyzer'],                                 
                                max_df=options['max_df'], min_df=options['min_df'],                                      
                              max_features=options['max_features'], binary = options['binary'])
        else:
            #TfidfVectorizer
            self.vectorizer = TfidfVectorizer(strip_accents=options['strip_accents'], stop_words=options['stop_words'], 
                              ngram_range=options['ngram_range'], analyzer=options['analyzer'],   max_df=options['max_df'], min_df=options['min_df'],                                                    
                              max_features=options['max_features'], binary = options['binary'],norm = options['norm'],sublinear_tf = opetions['sublinear_tf']) 
        self.vectorizer.fit(documents)
    
    def get_text_features(self,texts):
        return self.vectorizer.transform(texts)
        
if __name__ == '__main__':
    documents = ['I am a test string','I am another test string','and a third test string']
    small_docs = ['I am a test string','I am another test string']
    A =  TextFeaturizer(documents)#Count Vectorizer
    X = A.get_text_features(small_docs)
    B =  TextFeaturizer(documents,options=BASE_OPTIONS_TFIDF,Count=False)#TFIDF Vectorizer
    X = B.get_text_features(small_docs)

    
