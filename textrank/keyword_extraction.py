import editdistance
import itertools
import networkx
import nltk
from .ordered_set import OrderedSet
import re

def filter_tags(tags):
  # we don't like articles etc as a keyword
  # thats why we filter out only the relevant words
  allowed = ["JJ", "NN", "NNP"]
  # we also filter tags which don't contain letters at all, or are simply too short
  r=re.compile(r"[a-zA-Z]")
  return [i for i in tags if i[1] in allowed and r.search(i[0]) and len(i[0]) > 3]

def uniquify(tags):
  # each tag will only be returned once
  ret = OrderedSet()
  
  for t in tags:
    if t not in ret:
      ret.add(t)
  
  return ret

def build_graph(words):
  # returns a graph with the given words connected to each other
  # uses levenshtein distance to measure the distance between words
  
  graph = networkx.Graph()
  graph.add_nodes_from(words)

  pairs = itertools.combinations(words, 2)

  for p in pairs:
    dist = editdistance.eval(p[0], p[1])
    graph.add_edge(p[0], p[1], weight=dist)
  
  return graph

def extract_keywords(text):
  word_tokens = nltk.word_tokenize(text)
  pos_tagged = nltk.pos_tag(word_tokens)
  pos_tagged = filter_tags(pos_tagged)
  unique_tags = uniquify(pos_tagged)
  graph = build_graph(unique_tags)
  page_rank = networkx.pagerank(graph, weight="weight")
  keywords = sorted(page_rank, key=page_rank.get, reverse=True)

  # amount of keywords will be one third of all unique keywords
  amount = len(unique_tags) / 3
  keywords = keywords[0:amount+1]
  
  return [k[0] for k in keywords]
