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
  return [i for i in tags if i[1] in allowed and r.search(i[0])]

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

def extract_keywords(text, max=20):
  # extract max keywords for the given text

  word_tokens = nltk.word_tokenize(text)
  pos_tagged = nltk.pos_tag(word_tokens)
  pos_tagged = filter_tags(pos_tagged)
  unique_tags = uniquify(pos_tagged)
  graph = build_graph(unique_tags)
  page_rank = networkx.pagerank(graph, weight="weight")
  keywords = [k[0] for k in sorted(page_rank, key=page_rank.get, reverse=True)]

  # amount of keywords will be one third of all unique keywords
  amount = len(unique_tags) / 3
  if amount > max:
    amount = max
  
  keywords = keywords[0:amount+1]

  final_keyphrases = []
  
  processed = []
  
  ui = unique_tags.pop(False)[0]

  while unique_tags:
    uj = unique_tags.pop(False)[0]
    if ui in keywords and uj in keywords:
      final_keyphrases.append(ui + " " + uj)
      processed.append(ui)
      processed.append(uj)
    else:
      if ui in keywords and not ui in processed:
        final_keyphrases.append(ui)
      if not unique_tags and uj in keywords and uj not in processed:
        final_keyphrases.append(uj)
    ui = uj

  return final_keyphrases
