# //////////////////////////////////////////////////////////
#   question_classifier.py
#   the "wc" command
#   Created on:      11-Oct-2021 12:55:00
#   Original Author: J. Sayce
#   Specification:
#
#   question_classifier.py is the base file for the Team Late 
#   submission for the Question Classifier project. It:
#
#   - Handles arguments
#   - Validates config files (and complains if they're wrong)
#   - Loads appropriate modules (preprocessor, embedder...) into generic steps
#   - Conducts generic model behaviours on module instances
#   - Aggregates ensemble model results (if appropriate)
#   - Handles formatting of feedback/errors from modules to user
#
# //////////////////////////////////////////////////////////

# Library Imports
import random
import re
import time
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import random_split, DataLoader
from sklearn.metrics import classification_report
import numpy as np
import configparser
import argparse

# Local Imports
from bow import main as bow_main
from bilstm import main as bilstm_main
from embedding import main as embedding_main
from ffnn_classifier import main as ffnn_main
from question_dataset import QuestionDataset

# Basic Structural Definitions
model_sources = [['bow', bow_main], ['bilstm', bilstm_main]]
model_dict = {'bow': bow_main, 'bilstm': bilstm_main}

torch.manual_seed(1)
random.seed(1)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def main():
    # Read Arguments
    args = handle_arguments()

    # Read Config Files
    config = read_config(args.config)

    ensemble_size = int(config["Model"]["ensemble_size"])

    # Retrieve
    if args.train:
        data = load_data(config["Paths"]["path_train"])

        stop_words = load_data(config["Paths"]["stop_words"])

        # Tokenize and gen. word embeddings (RandomInit, Pre-trained), if "train" arg specified
        vocab_list = tokenise_data(data, stop_words, int(config["Model"]["vocab_min_occurrence"]))

        word_embedding, vocab_list = embedding_main(vocab_list, config)

        encode_data, labels = encode_sentence(data, vocab_list, stop_words, int(config["Model"]["sentence_max_length"]))
        classes = list(set(labels))
        classes.sort()
        indexed_label = [classes.index(x) for x in labels]

        for i in range(ensemble_size):
            # Train selected model (BOW or BiLSTM) if "train" arg specified
            if ensemble_size > 1:
                sub_idx = np.random.choice(range(len(encode_data)), size=int(len(encode_data) / ensemble_size))
                print(len(sub_idx))
                X_sub = [encode_data[i] for i in sub_idx]
                y_sub = [indexed_label[i] for i in sub_idx]
            else:
                X_sub = encode_data
                y_sub = indexed_label

            dataset = QuestionDataset(X_sub, y_sub)
            train_ds, val_ds = random_split(dataset, [len(X_sub) - int(len(X_sub) * 0.1), int(len(X_sub) * 0.1)])
            batch_size = int(config["Network Structure"]["batch_size"])
            train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
            val_dl = DataLoader(val_ds, batch_size=batch_size)

            model_fn = model_dict[config["Model"]["model"]]
            model = model_fn(word_embedding, read_config(config["Paths"]["%s_config" % (config["Model"]["model"])]),
                             len(classes))
            model.to(device)

            model, optimiser = train_model(model, train_dl, val_dl,
                                           epochs=int(config["Model Settings"]["epoch"]),
                                           lr=float(config["Hyperparameters"]["lr_param"]),
                                           opt_fn=torch.optim.SGD,
                                           loss_fn=F.cross_entropy)

            torch.save({
                "model_state_dict": model.state_dict(),
                # "optimizer_state_dict": optimiser.state_dict()
            }, "{0}model.{1}.{2}.pt".format(config["Paths"]["path_model_prefix"], config["Model"]["model"], i))

        torch.save({
            "word_embedding_state_dict": word_embedding.state_dict(),
            "stop_words": stop_words,
            "vocab_list": vocab_list,
            "classes": classes
        }, config["Paths"]["path_cache"])

    elif args.test:
        checkpoint = torch.load(config["Paths"]["path_cache"])

        stop_words = checkpoint["stop_words"]
        vocab_list = checkpoint["vocab_list"]
        classes = checkpoint["classes"]

        word_embedding = nn.Embedding(checkpoint["word_embedding_state_dict"]["weight"].size(dim=0),
                                      checkpoint["word_embedding_state_dict"]["weight"].size(dim=1))
        word_embedding.load_state_dict(checkpoint["word_embedding_state_dict"])

        results = []

        test_data = load_data(config["Paths"]["path_test"])
        encode_data, labels = encode_sentence(test_data, vocab_list, stop_words, int(config["Model"]["sentence_max_length"]))
        # test_ds = QuestionDataset(encode_data, [classes.index(x) for x in labels])
        # test_dl = DataLoader(test_ds)

        indexed_label = torch.LongTensor([classes.index(x) for x in labels])

        for i in range(ensemble_size):
            checkpoint = torch.load("{0}model.{1}.{2}.pt".format(config["Paths"]["path_model_prefix"], config["Model"]["model"], i))
            model_fn = model_dict[config["Model"]["model"]]
            model = model_fn(word_embedding, read_config(config["Paths"]["%s_config" % (config["Model"]["model"])]),
                             len(classes))
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(device)

            results.append(test_model(model, torch.LongTensor(encode_data), indexed_label, classes))

        y = indexed_label
        results = torch.stack(results)
        y_pred, _ = torch.mode(results, 0)

        # y = y.cpu()
        # y_pred = y_pred.cpu()
        idx_list = torch.ones(len(classes))
        for n in y.unique():
            idx_list[n.item()] = 0
        for n in y_pred.unique():
            idx_list[n.item()] = 0
        target_names = classes.copy()
        for i in torch.flip(torch.nonzero(idx_list, as_tuple=False), dims=[0]).squeeze():
            del target_names[i]
        print(classification_report(y.cpu(), y_pred.cpu(), target_names=target_names, zero_division=0))

    #         # Classify data (accuracy/F1 scores) produced by model if "test" arg specified
    #         results.append(classify_model_output())

    # aggregate_results(results)
    #
    # display_results()


def handle_arguments():
    # check parsed arguments (as found in coursework pdf)
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Configuration file')
    parser.add_argument('--train', action='store_true', help='Training mode - model is saved')
    parser.add_argument('--test', action='store_true', help='Testing mode - needs a model to load')

    # file arguments
    args = parser.parse_args()
    return args


def read_config(config_file):
    # load config file (as found in coursework pdf)
    config = configparser.ConfigParser()
    config.sections()
    config.read(config_file)

    return config


def write_config(config_file, data):
    config = configparser.ConfigParser()

    with open(config_file, 'w') as file:
        config.write(file)


# Attempts to load data from the config-specified source for "Training Set 5".
def load_data(directory):
    data = []
    with open(directory, "r") as f:  # get data
        for line in f:
            data.append(line.strip())

    return data


# Split the data into tokens.
def tokenise_data(data, stop_words, min_occurence):
    vocab_counter = Counter()
    for line in data:
        [_, sentence] = line.split(" ", 1)
        words = sentence.strip().lower().split(" ")
        for word in words:
            if re.match("[^a-z]*[a-z]+[^a-z]*", word) \
                    and word not in stop_words:
                vocab_counter.update([word])

    vocab_list = []
    for word, count in vocab_counter.items():
        if count >= min_occurence:
            vocab_list.append(word)

    vocab_list.append("#unk#")

    return vocab_list


def encode_sentence(sentence_list, vocab_list, stop_words, max_len):
    X = []
    y = []
    unk_idx = vocab_list.index("#unk#")
    for sentence in sentence_list:
        [label, sentence] = sentence.split(" ", 1)
        words = sentence.strip().lower().split(" ")
        seq = [0] * max_len
        i = 0
        for word in words:
            if i == max_len:
                break
            # valid word
            if re.match("[^a-z]*[a-z]+[^a-z]*", word):
                # skip stop words
                if word in stop_words:
                    continue
                try:
                    seq[i] = vocab_list.index(word)
                except ValueError:
                    seq[i] = unk_idx
                i += 1
        X.append(seq)
        y.append(label.lower().strip())
    return X, y


# Calls external model (as specified by config) with data, receives returned data, saves results.
def train_model(model, train_dl, val_dl, epochs=10, lr=0.1, opt_fn=torch.optim.SGD, loss_fn=F.cross_entropy):
    optimiser = opt_fn(filter(lambda e: e.requires_grad, model.parameters()), lr=lr)
    for epoch in range(epochs):
        start_time = time.time()

        # Training phase
        model.train()
        train_total = 0
        total_train_loss = 0.0
        for x_batch, y_batch in train_dl:
            # Forward pass and get prediction
            x_batch = x_batch.type(torch.LongTensor).to(device)
            y_batch = y_batch.to(device)
            y_out = model(x_batch)

            # Compute the loss, gradients, and update parameters
            optimiser.zero_grad()
            loss = loss_fn(y_out, y_batch)
            loss.backward()
            optimiser.step()

            # Accumulate loss
            train_total += y_batch.shape[0]
            total_train_loss += loss.item() * y_batch.shape[0]
        train_loss = total_train_loss / train_total

        # Validation phase
        model.eval()
        correct = 0
        val_total = 0
        total_val_loss = 0.0
        for x_batch, y_batch in val_dl:
            # print(x_batch)
            x_batch = x_batch.type(torch.LongTensor).to(device)
            y_batch = y_batch.to(device)
            y_out = model(x_batch)
            loss = loss_fn(y_out, y_batch)
            val_total += y_batch.shape[0]
            total_val_loss += loss.item() * y_batch.shape[0]
            # _, y_pred = torch.max(y_out, 1)
            # print(y_out)
            y_pred = torch.max(y_out, 1)[1]
            correct += (y_pred == y_batch).sum().item()
            # print("out")
            # print(y_pred, y_batch)
            # print(y_pred)
            # print(y_batch)
            # print((y_pred == y_batch), (y_pred == y_batch).sum())
        val_loss = total_val_loss / val_total
        val_acc = correct / val_total
        # print(correct, val_total)

        elapsed_time = time.time() - start_time

        print('Epoch [{:02}/{:02}] \t loss={:.4f} \t val_loss={:.4f} \t val_acc={:.4f} \t time={:.2f}s'.format(
            epoch + 1, epochs, train_loss, val_loss, val_acc, elapsed_time))

    return model, optimiser


# Attempts to run BOW or BiLSTM with data, receives returned data, and saves results.
def test_model(model, x, y, classes):
    x = x.to(device)
    y = y.to(device)
    y_out = model(x)
    y_pred = torch.max(y_out, 1)[1]
    # correct = (y_pred == y).sum().item()
    # accuracy = correct / y_out.shape[0]
    # print(y)
    # print(y_pred)
    return y_pred


# Attempts to run FF-NN with data, receives returned data, and saves results.
def classify_model_output():
    print("Debug!")


def aggregate_results():
    print("Debug!")


def display_results():
    print("Debug!")


if __name__ == "__main__":
    main()
