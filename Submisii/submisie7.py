import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neural_network import MLPClassifier

train_samples = []
train_labels = []
validation_samples = []
validation_labels = []
test_samples = []
test_ids = []
vocab = set()


def read_from_file(path, my_list, type='train'):
    f = open(path, "r", encoding='utf8')
    line = f.readline()
    while line != "":
        id, text = line.split("\t", 1)
        if type == "test":
            test_ids.append(id)
            text = text.strip()
            my_list.append(text)
        elif type == 'train':
            text = text.strip()
            my_list.append(text)
        elif type == 'label':
            text = text.strip()
            my_list.append(int(text))
        line = f.readline()
    f.close()
    my_list = np.array(my_list)
    return my_list


def create_vocab(my_list):
    global vocab
    for text in my_list:
        for word in text.split(" "):
            vocab.add(word)
    vocab = list(vocab)


if __name__ == '__main__':
    # read train samples
    train_samples = read_from_file("../input/data-aliens/train_samples.txt", [])

    # read train labels
    train_labels = read_from_file("../input/data-aliens/train_labels.txt", [], 'label')

    # read validation samples
    validation_samples = read_from_file("../input/data-aliens/validation_samples.txt", [])

    # read validation labels
    validation_labels = read_from_file("../input/data-aliens/validation_labels.txt", [], 'label')

    # read test samples
    test_samples = read_from_file("../input/data-aliens/test_samples.txt", [], "test")

    # find all words in train and validation files to get an idea of what the samples look like
    create_vocab(np.concatenate((train_samples, validation_samples)))
    f = open("./informations.txt", "w", encoding='utf8')
    f.write("There are " + str(len(vocab)) + " distinct words in train+validation\n")
    f.close()

    # use the best parameters from a previous search, named 'NN n-gram on chars parameter search.py'
    vect = CountVectorizer(analyzer='char', lowercase=False, ngram_range=(5,5))

    train_data = vect.fit_transform(train_samples)
    validation_data = vect.transform(validation_samples)

    model = MLPClassifier(hidden_layer_sizes=(32,), early_stopping=True, learning_rate_init=0.0001)

    model.fit(train_data, train_labels)
    predictions = model.predict(validation_data)
    accuracy = np.mean(predictions == validation_labels)

    path = "./predictions.txt"
    f = open(path, "w", encoding='utf8')
    f.write("Correct, wrong")
    for i in range(predictions.size):
        f.write("\n" + str(validation_labels[i]) + " " + str(predictions[i]))
    f.close()
    
    f = open("./informations.txt", "a", encoding='utf8')
    f.write("Accuracy: " + str(accuracy) + "\n") 
    f.close()   

    no_epoch = model.n_iter_
    best_model = MLPClassifier(hidden_layer_sizes=(32,), early_stopping=False, learning_rate_init=0.0001, max_iter=no_epoch)
    train_and_validation_data = vect.fit_transform(np.concatenate((train_samples, validation_samples)))
    best_model.fit(train_and_validation_data, np.concatenate((train_labels, validation_labels)))

    test_data = vect.transform(test_samples)
    predictions = best_model.predict(test_data)

    f = open("./test_labels_NN_NGRAM.txt", "w", encoding='utf8')
    f.write("id,label")
    for i in range(predictions.size):
        f.write("\n" + test_ids[i] + "," + str(predictions[i]))
    f.close()