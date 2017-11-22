# Source files:
1. graph_generators.py 

# Setup:
1. I have defined the paths to the given datasets in a list called GRAPHS as follows:

- USairport_2010 = "data/USairport_2010.elist.txt"
- imdb_actors = "data/imdb_actor.elist.txt"

Please update the path as desired.

2. Calculating some of the metrics on larger networks takes a long time. Please comment/uncomment 
the desired metrics or graphs on which they are calculated as desired.

3. Packages used:
- networkx 2.0
- snap
- matplotlib

# Usage:
The script takes in 0 or 1 arguments. With 0 arguments, the script generates all the graphs.
The argument specifies whether to calculate metrics on the graphs or to plot the degree distributions.

python graph_generators.py [ plt | metrics ]
