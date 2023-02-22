import numpy as np
from sklearn.svm import SVC

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
            test_ids.append(id) # save the test ids for predictions
            text = text.strip()
            my_list.append(text.split(" "))
        elif type == 'train':
            text = text.strip()
            my_list.append(text.split(" "))
        elif type == 'label':
            text = text.strip()
            my_list.append(int(text))
        line = f.readline()
    f.close()
    my_list = np.array(my_list)
    return my_list

def create_vocab(my_list):
    global vocab
    # find all distinct words from the file
    for text in my_list:
        for word in text:
            vocab.add(word)
    vocab = list(vocab)


def create_features(my_list):
    # in freq_matrix:
    # each row corresponds to a line of text from train_samples
    # each column corresponds to a word from the vocabulary
    # each value is the frequency of that word in the current text/line
    global vocab
    freq_matrix = np.array([[text.count(word) for word in vocab] for text in my_list])

    return freq_matrix

if __name__ == '__main__':
    # read train samples
    train_samples = read_from_file("./data-aliens/train_samples.txt", [])

    # read train labels
    train_labels = read_from_file("./data-aliens/train_labels.txt", [], 'label')

    # read validation samples
    validation_samples = read_from_file("./data-aliens/validation_samples.txt", [])

    # read validation labels
    validation_labels = read_from_file("./data-aliens/validation_labels.txt", [], 'label')

    # read test samples
    test_samples = read_from_file("./data-aliens/test_samples.txt", [], "test")

    # find all distinct words from the train samples
    create_vocab(train_samples)

    training_data = create_features(train_samples)
    validation_data = create_features(validation_samples)

    # try BOW on SVM model
    model = SVC(decision_function_shape="ovo")
    model.fit(training_data, train_labels)
    predictions = model.predict(validation_data)
    accuracy = np.mean(predictions == validation_labels)

    print("accuracy:", accuracy)
    f = open('./predictions.txt', "w", encoding='utf8')
    f.write("Correct, wrong")
    for i in range(predictions.size):
        f.write("\n" + str(validation_labels[i]) + " " + str(predictions[i]))
    f.close()

    test_data = create_features(test_samples)
    predictions = model.predict(test_data)

    f = open("./data-aliens/test_labels.txt", "w", encoding='utf8')
    f.write("id,label")
    for i in range(predictions.size):
        f.write("\n" + test_ids[i] + "," + str(predictions[i]))
    f.close()