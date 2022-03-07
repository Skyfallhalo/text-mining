import random 

random.seed(1)

with open("train_5500.label.txt", "r") as f: # get data
   train = f.readlines()

random.shuffle(train) # random shuffle

dev_data = train[:int(len(train)/10)] # dev split
train_data = train[int(len(train)/10):] # train split

with open('dev.txt', 'w') as dev: # write to dev

   for i in range(len(dev_data)):

      dev.write(dev_data[i])

with open('train.txt', 'w') as train: # write to train

   for i in range(len(train_data)):

      train.write(train_data[i])


with open("TREC_10.label.txt", "r") as f: # get data
   test_data = f.readlines()

with open('test.txt', 'w') as test: # write to train

   for i in range(len(test_data)):

      test.write(test_data[i])