#!/bin/bash

for f in data/American_recipes.elist.txt data/South-America_recipes.elist.txt data/India_recipes.elist.txt data/Canada-Scandinavia.elist.txt
do
#    time python analyze-centrality.py $f;
#    time python analyze-centrality.py $f -plt;
#    time python analyze-centrality.py $f -scatter;

    time python ../assignment1/gen-structure.py $f

#    time python analyze-centrality.py $f

#    time python analyze-connectivity.py $f
done
