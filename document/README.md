question_classifier.py

Created on:      01-Mar-2022 12:55:00
Original Authors: Alexander Trebilcock, Ashka K. Abrarriadi, Joshua Sayce, Oluwadamini Akpotohwo, Wenqi Han, Zhaoyu Han
Specification:


Overview

This program is a hybrid question classifier utilising supervised machine learning methods, developed by Team Late. It attempts to classify plaintext english question statements, such as "What year was Disneyland constructed?", to their predicted response type ("NUM:date").

The classifier can be configured to use two models: 

    Bag-of-Words (a data-driven approach, which takes the form of a binary-encoded vector)

    BiLSTM (A recurrent neural network which can model long-term dependencies through a memory vector).

The program begins by reading a specified configuration file for locations of training, or test data (in the form of newline-delimited question/label pairs.) It splits these into two lists, respectively.

As part of preprocessing, the question set split into tokens, stripped of 'stopwords' (pre-defined words of little use) converted into lemma form, and a unique token list is derived from it. 

Word embeddings are generated from the unique token words derived earlier. These are vectors assigned to each token containing 'weights', which can be compared to determine semantic similarities between words. Two approaches are available to the program:

    Randomly-initialised embeddings: Initially, no word is given special relationship with any other.
    
    Pre-trained embeddings: An external corpus is referenced to give each unique token a meaningful weight.

In the latter case, a local configurable file is referenced, and used as a lookup against the unique token list provided.

Finally, the data must be encoded - or converted from tokenised strings into the indexes of their locations within the unique token list.

A model is constructed, initialised using the generated embeddings, and given a reference to a config file, which it will use to self-configure the number of layers, hidden layers, and dimensions of the output classes.


Training

The model can then be trained if the correct command was supplied - test data is input in 'batches', or subsets of encoded texts and label pairs. The model will use the embedding matrix passed earlier to convert the batch into a distributional representation. The prediction network (with initially random weights) will then be applied, classifying the results of the encoded text's network traversal using a softmax activation function. 

The output labels/classifications of the model are then compared against the "gold standard" labels using a "loss function" (which uses the outputs of the model to evaluate the epoch's accuracy) - this outputs gradients of the loss. An optimzer (SGD) is then used to feed the gradients back into the NN - "backpropogating" the weights into the model to minimise loss for the next epoch. Followng this step, in the training loop, these gradients are not accumulated, they are "zeroed" each epoch.

This process is repeated a configurable number of times (or 'epochs'), with the optimal weights backpropogated into the model each time. After a set number of epochs, the final model weights are saved.

If an "ensemble" number is specified, the entire training process is repeated for several distinct instances of model - all of which generate unique outputs. The outputs of each classification are combined into an optimised result.


Testing

The model can be tested if the correct command is supplied. The process is similar to the training method, but loss gradients are accumulated each epoch. The results of classification (i.e. label predictions) are returned.


Outputs

Following the ensemble loop, the program will amalgamate the ensemble results into an average. In test mode, a file will be generated displaying the predicted class for each of the inputs.


How to Use

The program can be ran by calling "question_classifier.py" on the python command line:

    #python3 question_classifier.py

A single mandatory argument must be passed, --train or --test, which informs the classifier to run in the referenced "mode" (note: a model must be trained prior to testing, or the user will be warned that the testing model is invalid). 

All additional parameters may be found in the configuration file (default data/config.ini). A custom configuration file can be specified by using the --config flag, and supplying a directory to source the file.


Argument Listing:

    --train : Mandatory exclusive argument. Signals the classifier to run in training mode.
    
    --test : Mandatory exclusive argument. Signals the classifier to run in testing mode.
    
    --config [configuration_file_path] : Optional argument. Specify a custom configuration file.


Directory and File Listing:

data/ 
A directory that holds input and output files to be used by the program.

    data/bilstm_config.ini
    Configuration file for the BiLSTM model, containing parameters to be applied to the nn.Model instance.
    
    data/bow_config.ini
    Configuration file for the BOW model, containing parameters to be applied to the nn.Model instance.
    
    data/classifier_config.ini
    Configuration file for the training and testing methods for the supplied model, such as hyperparameters.
    
    data/config.ini
    The master configuration file, containing paths, and settings too generic to go into a specific config file (such as embeddings).
    
    data/stopWords.txt
    A pre-supplied list of "stopwords", which are omitted in the tokenisation process.

document/
A directory holding supplementary documentation for the program.
    
    document/README.md
    An overview of the program's purpose, high-level behaviour, and a series of instructions on how to use the program.
    
    document/program_listing.md
    Documentation on the functions within the program (and constituent source files). 
    
    document/requirements.txt
    A list of non-version-specific packages required by the program.
    
src/
A directory containing the source code of the program.
    
    src/bilstm.py
    A file containing a derivation of torch.nn.Model, configured for BiLSTM. Specifies a forward pass function.
    
    src/bow.py
    A file containing a derivation of torch.nn.Model, configured for BOW. Specifies a forward pass function.
    
    src/embedding.py
    A file containing both embedding methods for inputted data - pretrained (a local file sourced from GloVe), and random. Regardless of method, instantiates and returns a pytorch.nn.Embedding, and trimmed vocabulary list, given a passed list of unique vocabulary.
    
    src/ffnn_classifier.py
    A file containing the training and test functions used to operate on the model instance created (BiLSTM/BOW). Passed the model, encoded dataset, and true labels. Trains/tests model over epochs, taking batches of the dataset, passing them into the NN, and backpropogating resulting loss gradients through an optimiser. Returns the 'best' parameters (those that result in the least loss).
    
    src/question_classifier.py
    The main program file. Controls the logic of the classifier; reads arguments, imports the external embedding/model/train/test functions, and defines the data pipeline between them.
    
    
Configuration Parameter Listing: