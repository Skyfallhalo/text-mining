# Program Listing

As most functions are called in question_classifier.py. Those functions will be introduced here.

**handleArguments**: Checks for the three required arguments - train or test, manually specify config, and config path.

**readConfig**: Attempts to parse provided file as config.

**loadData**: Attempts to load data from the config-specified source for "Training Set 5".

**tokeniseData**: Split the data into tokens. Returns tokenised text strings, and dict of unique words.

**preprocessData**: Splits data into strings, and their respective labels.

**generateWordEmbeddings**: Container for link to embedding.py's embedding methods. Returns embeddings, and vocab list.

**encodeData**: Converts data into their indexes of their vocabulary, and pads if appropriate.

**generateDatasets**: Generate training and development datasets for training model

**trainModel**: Contains code for training

**testModel**: Contains code for testing

**classifyModelOutput**: Attempts to run FF-NN with data, recieves returned data, and saves results.

**aggregateResults**: Takes the map of tensor results, and uses an ensemble model to generate a single set of classifications.

**outputResults**: Output the testing result to a text file