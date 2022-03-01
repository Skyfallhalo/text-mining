#//////////////////////////////////////////////////////////
#   ffnn_classifier.py
#   Created on:      01-Mar-2022 14:30:00
#   Original Author: J. Sayce
#   Specification:
#
#   base file for the feed-forward neural network classifier,
#   used by used by question_classifier.py.
#
#//////////////////////////////////////////////////////////

def main(*args):
    readConfig(args)
    
#Input: Config directory passed from question_classifier.py
#Task: Populate config values by reading config.ini
#Output: config.
def readConfig(configFile): 
    print("Debug")    
    
if __name__ == "__main__":
    main(*sys.argv[1:])
    sys.exit(0)
