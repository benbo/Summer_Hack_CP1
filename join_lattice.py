from __future__ import print_function
from load_data import load_gzip_field
import progressbar
import os,sys
import gzip as gz
import json
import codecs

def find_match(iterator,lookup):
    for line in iterator:
        d = json.loads(line)
        if d['_id'] in lookup:
            yield line



if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-infile','-i',required=True)
    parser.add_argument('-lattice_root','-l',required=True)
    parser.add_argument('--outfile', '-o',default='lattice_matches')
    args = parser.parse_args()
    outfile = args.outfile
    if not '.gz' in outfile:
        outfile +='.gz'
    lattice_root = args.lattice_root
    filename = args.infile
    #load CDR ids of labeled data
    ids = frozenset([i for i in load_gzip_field(file_names=[args.infile],field='doc_id')])
    #iterate through lattice extraction files, find matching ids
    bar = progressbar.ProgressBar(max_value=progressbar.UnknownLength)
    if not os.path.exists('out'):
        os.makedirs('out')
    match_count = 0
    search_count = 0
    with gz.GzipFile(os.path.join('out',outfile),'w') as outf:
        for root, dirs, files in os.walk(lattice_root, topdown=False):
            for name in files:
                if 'data' in name:
                    if 'gz' in name:
                        with gz.GzipFile(os.path.join(root, name),'r') as inf:
                            try:
                                matches = tuple(d for d in find_match(inf,ids)) 
                            except IOError:
                                print('failed to open file: {}'.format(inf))
                                continue
                            for m in matches:
                                outf.write(m+'\n')
                            match_count +=len(matches)
                        search_count += 1
                        bar.update(match_count)
