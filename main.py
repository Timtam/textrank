from textrank.keyword_extraction import extract_keywords
from nltk.corpus import PlaintextCorpusReader
import os.path
import random

path = os.path.join(os.path.abspath(os.path.dirname(__file__)), "citeulike180", "documents")

print("Parsing corpus from path {0}".format(path))

corpus = PlaintextCorpusReader(path, ".*", encoding="latin-1")

files = corpus.fileids()

print("Found {0} files within corpus.".format(len(files)))

fileid = random.randint(0, len(files))

file = files[fileid]

print("Randomly selected file {0} for processing.".format(file))

print("Extracting keywords...")

print(extract_keywords(corpus.raw(file)))
