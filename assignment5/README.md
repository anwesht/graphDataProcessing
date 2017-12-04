# Source files:
1. info_dissemination.py 

# Setup:
1. Input graphs for the experiments are in the folder data.

2. Packages used:
- networkx 2.0
- matplotlib
- numpy
- abc 
- pprint

# Usage:
The script takes in 0 or 1 arguments. With 0 arguments, the script runs all the experiments. It excepts certain 
files in the data folder to exist. These can be generated using the 1 argument this script takes.

python info_dissemination.py [ gen ]

# Other files/folders:
1. data -> Contains the graphs used for experiments
2. output -> Contains the output created by the script.
3. experiments -> contains the results of running the experiments i.e. plots and generated dynamic graphs.
4. videos -> screen captures for visualization of the dynamic graphs on Gephi.
