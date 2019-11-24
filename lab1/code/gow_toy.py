import string
from nltk.corpus import stopwords
from igraph import *

#import os
#os.chdir() # to change working directory to where functions live
# import custom functions
from library import clean_text_simple, terms_to_graph, core_dec

stpwds = stopwords.words('english')
punct = string.punctuation.replace('-', '')

my_doc = 'A method for solution of systems of linear algebraic equations \
with m-dimensional lambda matrices. A system of linear algebraic \
equations with m-dimensional lambda matrices is considered. \
The proposed method of searching for the solution of this from library import clean_text_simple, system \
lies in reducing it to a numerical system of a special kind.'

# my_doc = 'graph text mining density graph text mining'
my_doc = my_doc.replace('\n', '')

# pre-process document
my_tokens = clean_text_simple(my_doc,my_stopwords=stpwds,punct=punct)

g = terms_to_graph(my_tokens, 4)

# number of edges
print(len(g.es))

# the number of nodes should be equal to the number of unique terms
len(g.vs) == len(set(my_tokens))

edge_weights = []
for edge in g.es:
    source = g.vs[edge.source]['name']
    target = g.vs[edge.target]['name']
    weight = edge['weight']
    edge_weights.append([source, target, weight])

print(edge_weights)
layout = g.layout("kk")
visual_style = {}
visual_style["vertex_size"] = 20
visual_style["vertex_label"] = g.vs["name"]
visual_style["edge_width"] = [1 + 2 * int(is_formal) for is_formal in g.es['weight']]
visual_style["layout"] = layout
visual_style["bbox"] = (300, 300)
visual_style["margin"] = 20
plot(g, **visual_style)

for w in range(2,10):
    g = terms_to_graph(my_tokens, w)
    ## fill the gap (print density of g) ###
    print('The density with a window of size {} is: {}'.format( w, g.density()))

# decompose g
g = terms_to_graph(my_tokens, 4)
core_numbers = core_dec(g,False)
print(core_numbers)

### fill the gap (compare 'core_numbers' with the output of the .coreness() igraph method) ###
print(g.coreness())
# retain main core as keywords
max_c_n = max(core_numbers.values())
keywords = [kwd for kwd, c_n in core_numbers.items() if c_n == max_c_n]
print(keywords)
