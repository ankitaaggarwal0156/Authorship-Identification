#!/usr/bin/env python
from collections import defaultdict
from csv import DictReader, DictWriter
import re
import nltk
import codecs
import sys
from nltk.corpus import wordnet as wn
from nltk.tokenize import TreebankWordTokenizer
from string import punctuation


kTOKENIZER = TreebankWordTokenizer()

def morphy_stem(word):
    """
    Simple stemmer
    """
    stem = wn.morphy(word)
    if stem:
        return stem.lower()
    else:
        return word.lower()

class FeatureExtractor:
    def __init__(self):
        """
        You may want to add code here
        """
        self._cmuDict =  nltk.corpus.cmudict.dict()
        self._d1 = self.ext_feature()
        
        None

    def features(self, text):
        d = defaultdict(int)
        for ii in kTOKENIZER.tokenize(text):
            if morphy_stem(ii) in self._cmuDict:
                d[morphy_stem(ii)] += 1
            else:
                if ii not in punctuation:
                    d[morphy_stem(ii)] += 2
        a, punc_norm= self.parseStressOfLine(text)
        for i in punc_norm:
            d[i]+=1
        if len(a)>9:
            d['Iambic_pentameter'] = 1
        else:
            d['Iambic_pentameter'] = 0
        
        d['word_no_stress']= a
        d['num_of_syllable']= len(a)
        d['length_of_line'] = len(text)
        return d
    
    def sent_tokenize(self, text):
        #separate into 2 tokens, if a word consists of a dash
        text = text.replace("-"," ")
        items = text.split()
        tokens=list()
        for i in items:
            i= i.strip()
            if i=="":
                continue
            #replace all other punctuation marks
            word = re.sub('[^a-zA-Z ]', '', i)
            tokens.append(word.lower())
        return tokens
    
#Read external file Poetry from other author to make similarity with Shakespeare's writing style.    
    def ext_feature(self):
        text1=[]
        #d1 = []
        with open("./external.txt", "r") as fo:
            for line in fo:
                #line = line.strip()
                text1.append(line)
        fo.close()
         
        print "ggg", text1 
        #for i in text1:
         #   d1.append(self.parseStressOfLine(i))
        return text1
   
            
    
    def parseStressOfLine(self, line):

        stress=""
        tokens1 = [word for word in self.sent_tokenize(line)]
        tokens = [morphy_stem(words.lower()) for words in tokens1] 
        for word in tokens: 
            if word =='':
                continue
    
            if word not in self._cmuDict:
                # if word is not in dictionary, add the part that most resembles a dictionary word
                str=""
                for char in word:
                    str = str+ char
                    if str in self._cmuDict:
                        a = self.strip_letters(self._cmuDict[str][0])
                stress = stress+a    
            else:
                zero_bool=True
                for s in self._cmuDict[word]:
                    # search for a zero in array returned from cmudict
                    if self.strip_letters(s)=="0":
                        stress = stress + "0"
                        zero_bool=False
                        break
    
                if zero_bool:
                    stress = stress + self.strip_letters(self._cmuDict[word][0])
        return stress, tokens 

    # convert string of syllables to number
    def strip_letters(self,ls):
        nm = ''
        for ws in ls:
            for ch in list(ws):
                if ch.isdigit():
                    nm=nm+ch
        return nm  
        
        
reader = codecs.getreader('utf8')
writer = codecs.getwriter('utf8')


def prepfile(fh, code):
  if type(fh) is str:
    fh = open(fh, code)
  ret = gzip.open(fh.name, code if code.endswith("t") else code+"t") if fh.name.endswith(".gz") else fh
  if sys.version_info[0] == 2:
    if code.startswith('r'):
      ret = reader(fh)
    elif code.startswith('w'):
      ret = writer(fh)
    else:
      sys.stderr.write("I didn't understand code "+code+"\n")
      sys.exit(1)
  return ret

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument("--trainfile", "-i", nargs='?', type=argparse.FileType('r'), default=sys.stdin, help="input train file")
    parser.add_argument("--testfile", "-t", nargs='?', type=argparse.FileType('r'), default=None, help="input test file")
    parser.add_argument("--outfile", "-o", nargs='?', type=argparse.FileType('w'), default=sys.stdout, help="output file")
    parser.add_argument('--subsample', type=float, default=1.0,
                        help='subsample this fraction of total')
    args = parser.parse_args()
    trainfile = prepfile(args.trainfile, 'r')
    if args.testfile is not None:
        testfile = prepfile(args.testfile, 'r')
    else:
        testfile = None
    outfile = prepfile(args.outfile, 'w')

    # Create feature extractor (you may want to modify this)
    fe = FeatureExtractor()
    
    # Read in training data
    train = DictReader(trainfile, delimiter='\t')
    
    # Split off dev section
    dev_train = []
    dev_test = []
    full_train = []

    for ii in train:
        if args.subsample < 1.0 and int(ii['id']) % 100 > 100 * args.subsample:
            continue
        feat = fe.features(ii['text'])
        if int(ii['id']) % 5 == 0:
            dev_test.append((feat, ii['cat']))
        else:
            dev_train.append((feat, ii['cat']))
        full_train.append((feat, ii['cat']))

    # Train a classifier
    sys.stderr.write("Training classifier ...\n")
    classifier = nltk.classify.NaiveBayesClassifier.train(dev_train)

    right = 0
    total = len(dev_test)
    for ii in dev_test:
        prediction = classifier.classify(ii[0])
        if prediction == ii[1]:
            right += 1
    sys.stderr.write("Accuracy on dev: %f\n" % (float(right) / float(total)))

    if testfile is None:
        sys.stderr.write("No test file passed; stopping.\n")
    else:
        # Retrain on all data
        classifier = nltk.classify.NaiveBayesClassifier.train(dev_train + dev_test)

        # Read in test section
        test = {}
        for ii in DictReader(testfile, delimiter='\t'):
            test[ii['id']] = classifier.classify(fe.features(ii['text']))

        # Write predictions
        o = DictWriter(outfile, ['id', 'pred'])
        o.writeheader()
        for ii in sorted(test):
            o.writerow({'id': ii, 'pred': test[ii]})
