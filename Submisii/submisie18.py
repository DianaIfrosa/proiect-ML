import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neural_network import MLPClassifier

train_samples = []
train_labels = []
validation_samples = []
validation_labels = []
test_samples = []
test_ids = []

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

    # try n-grams of chars
    max_accuracy = 0
    best_epoch_number = 0
    best_learning_rate_init = 0
    best_hidden_layer_sizes = 0
    solution_index = 0
    best_solution_index = 0

    vect = CountVectorizer(analyzer='char', lowercase=False, ngram_range=(6,6))

    train_data = vect.fit_transform(train_samples)
    validation_data = vect.transform(validation_samples)

    for hidden_layer_sizes_param in [(32,64,128)]:
      for learning_rate_init_param in [0.0001, 0.00001]:
            solution_index += 1

            model = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes_param, early_stopping=True,
                                  batch_size=128, learning_rate_init=learning_rate_init_param)

            model.fit(train_data, train_labels)
            predictions = model.predict(validation_data)

            accuracy = np.mean(predictions == validation_labels)
            if accuracy > max_accuracy: # save the best parameters
                max_accuracy = accuracy
                best_epoch_number = model.n_iter_
                best_solution_index = solution_index
                best_learning_rate_init = learning_rate_init_param
                best_hidden_layer_sizes = hidden_layer_sizes_param
                
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
                    ", batch size 128" +
                    ": " + str(accuracy) + "\n")
            f.close()


    f = open("./informations.txt", "w", encoding='utf8')
    f.write("Best accuracy is for solution no. " + str(best_solution_index) + " hidden layers " + str(best_hidden_layer_sizes) +
                    ", learning rate " + str(best_learning_rate_init) +
                    ", epoch number " + str(best_epoch_number) +
                    ", batch size 128" +
                    ": " + str(max_accuracy) + "\n")
    f.close()

    best_model = MLPClassifier(hidden_layer_sizes=best_hidden_layer_sizes, early_stopping=False,
                               batch_size=128,learning_rate_init=best_learning_rate_init, max_iter=best_epoch_number)
    best_vect = CountVectorizer(analyzer='char', lowercase=False, ngram_range=(6,6))
    train_and_validation_data = best_vect.fit_transform(np.concatenate((train_samples, validation_samples)))
    best_model.fit(train_and_validation_data, np.concatenate((train_labels, validation_labels)))

    test_data = best_vect.transform(test_samples)
    predictions = best_model.predict(test_data)

    f = open("./test_labels_NN_NGRAM.txt", "w", encoding='utf8')
    f.write("id,label")
    for i in range(predictions.size):
        f.write("\n" + test_ids[i] + "," + str(predictions[i]))
    f.close()