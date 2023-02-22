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
    train_samples = read_from_file("./data-aliens/train_samples.txt", [])

    # read train labels
    train_labels = read_from_file("./data-aliens/train_labels.txt", [], 'label')

    # read validation samples
    validation_samples = read_from_file("./data-aliens/validation_samples.txt", [])

    # read validation labels
    validation_labels = read_from_file("./data-aliens/validation_labels.txt", [], 'label')

    # read test samples
    test_samples = read_from_file("./data-aliens/test_samples.txt", [], "test")

    # find all words in train and validation files to get an idea of what the samples look like
    create_vocab(np.concatenate((train_samples, validation_samples)))
    f = open("./informations.txt", "w", encoding='utf8')
    f.write("There are " + str(len(vocab)) + " distinct words in train+validation\n")
    f.close()
    
    # try n-grams on char (find the best n between [4, 8])
    n_options = [x for x in range(4,9)]

    max_accuracy = 0
    best_epoch_number = 0
    best_learning_rate_init = 0
    best_hidden_layer_sizes = 0
    best_batch_size = 0
    best_n = 0
    best_preprocessing = False
    solution_index = 0
    best_solution_index = 0

    for lowercase_preprocessing in [False,True]:
        for n in n_options:
            vect = CountVectorizer(analyzer='char', lowercase=lowercase_preprocessing, ngram_range=(n,n))

            train_data = vect.fit_transform(train_samples)
            validation_data = vect.transform(validation_samples)

            # try multiple variations of parameters
            for hidden_layer_sizes_param in [(32,), (32,64), (32,64,128)]:
                for learning_rate_init_param in [0.0001, 0.001, 0.01]:
                  for batch_size_param in [64, 32]:
                      solution_index += 1

                      model = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes_param, early_stopping=True,
                                            batch_size=batch_size_param, learning_rate_init=learning_rate_init_param)

                      model.fit(train_data, train_labels)
                      predictions = model.predict(validation_data)

                      accuracy = np.mean(predictions == validation_labels)
                      if accuracy > max_accuracy: #save the best parameters
                          max_accuracy = accuracy
                          best_epoch_number = model.n_iter_
                          best_n = n
                          best_solution_index = solution_index
                          best_preprocessing = lowercase_preprocessing
                          best_learning_rate_init = learning_rate_init_param
                          best_hidden_layer_sizes = hidden_layer_sizes_param
                          best_batch_size = batch_size_param
                          
                          path = "./predictions-NN-NGRAM-" + str(solution_index) + ".txt"
                          f = open(path, "w", encoding='utf8')
                          f.write("Correct, wrong")
                          for i in range(predictions.size):
                              f.write("\n" + str(validation_labels[i]) + " " + str(predictions[i]))
                          f.close()

                      f = open("./accuracy-NN-NGRAM.txt", "a", encoding='utf8')
                      f.write("Accuracy for hidden layers " + str(hidden_layer_sizes_param) +
                              ", learning rate " + str(learning_rate_init_param) +
                              ", epoch number " + str(model.n_iter_) +
                              ", n " + str(n) +
                              ", lowercase preprocessing " + str(lowercase_preprocessing) +
                              ", batch size " + str(batch_size_param) +
                              ": " + str(accuracy) + "\n")
                      f.close()


    f = open("./informations.txt", "a", encoding='utf8')
    f.write("Best accuracy is for solution no. " + str(best_solution_index) + " hidden layers " + str(best_hidden_layer_sizes) +
                    ", learning rate " + str(best_learning_rate_init) +
                    ", epoch number " + str(best_epoch_number) +
                    ", n " + str(best_n) +
                    ", lowercase preprocessing " + str(best_preprocessing) +
                    ", batch size " + str(best_batch_size) +
                    ": " + str(max_accuracy) + "\n")
    f.close()

    best_model = MLPClassifier(hidden_layer_sizes=best_hidden_layer_sizes, early_stopping=False,
                                learning_rate_init=best_learning_rate_init, max_iter=best_epoch_number)
    best_vect = CountVectorizer(analyzer='char', lowercase=best_preprocessing, ngram_range=(best_n,best_n))
    train_and_validation_data = best_vect.fit_transform(np.concatenate((train_samples, validation_samples)))
    best_model.fit(train_and_validation_data, np.concatenate((train_labels, validation_labels)))

    test_data = best_vect.transform(test_samples)
    predictions = best_model.predict(test_data)

    f = open("./test_labels_NN_NGRAM.txt", "w", encoding='utf8')
    f.write("id,label")
    for i in range(predictions.size):
        f.write("\n" + test_ids[i] + "," + str(predictions[i]))
    f.close()