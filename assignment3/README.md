# Source files:
1. generator.py => The implementation of the algorithm.
2. main.py => The script to perform the experiments and plot graphs.
3. analyzer.py => Returns a few required metrics on the graph.

# Setup:
1. I have defined the paths to the given datasets in main.py as follows:

- FACEBOOK = "./dataset/fb107.txt"
- ARXIV = "./dataset/caGrQc.txt"

Please follow this structure or update the path in these variables.

2. The facebook and arxiv datasets have two possible generators. One generates 
the graphs corresponding to p = [0, 1] and the other generates the graph with 
selected parameters. Please comment/uncomment the code in main.py as necessary.

# Usage:
The main.py script takes in 1 argument and generates the graphs and plots 
for the dataset corresponding to that argument.

python main.py facebook | arxiv | synth


 
