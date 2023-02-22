import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer

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
    train_samples = read_from_file("../input/data-aliens/train_samples.txt", [])

    # read train labels
    train_labels = read_from_file("../input/data-aliens/train_labels.txt", [], 'label')

    # read validation samples
    validation_samples = read_from_file("../input/data-aliens/validation_samples.txt", [])

    # read validation labels
    validation_labels = read_from_file("../input/data-aliens/validation_labels.txt", [], 'label')

    # read test samples
    test_samples = read_from_file("../input/data-aliens/test_samples.txt", [], "test")
    
    # try n-grams on chars (find the best n between [3, 20])
    n_options = [x for x in range(3,21)]
        
    # Naive Bayes with Multinomial distribution
    model = MultinomialNB() 
    max_accuracy = 0
    best_n = 0
    best_predictions = None
    best_preprocessing = None
        
    for n in n_options:
        for preprocessing in [True, False]:
            vect = CountVectorizer(analyzer='char', lowercase=preprocessing, ngram_range=(n,n))
            train_data = vect.fit_transform(train_samples)
            validation_data = vect.transform(validation_samples)

            model.fit(train_data, train_labels)

            predictions = model.predict(validation_data)

            accuracy = np.mean(predictions == validation_labels)
            if accuracy > max_accuracy: # save the best parameters
                max_accuracy = accuracy
                best_n = n
                best_predictions = predictions
                best_preprocessing = preprocessing

            f = open("./accuracy-NB-ngram.txt", "a", encoding='utf8')
            f.write("Accuracy for " + str(n) + "length: " + str(accuracy) + " preprocessing: " + str(preprocessing)+"\n")
            f.close()

    path = "./predictions.txt"
    f = open(path, "w", encoding='utf8')
    f.write("Correct, wrong")
    for i in range(best_predictions.size):
        f.write("\n" + str(validation_labels[i]) + " " + str(best_predictions[i]))
    f.close()
    f = open("./informations.txt", "a", encoding='utf8')
    f.write("Best accuracy: " + str(max_accuracy) + ", best n: " + str(best_n) + ", lowercase: " + str(best_preprocessing) + "\n") 
    f.close()   
    
    vect = CountVectorizer(analyzer='char', lowercase=best_preprocessing, ngram_range=(best_n,best_n)) 
    train_and_validation_data = vect.fit_transform(np.concatenate((train_samples, validation_samples)))
    model.fit(train_and_validation_data, np.concatenate((train_labels, validation_labels)))
    
    test_data = vect.transform(test_samples)
    predictions = model.predict(test_data)

    f = open("./test_labels_NB-ngram.txt", "w", encoding='utf8')
    f.write("id,label")
    for i in range(predictions.size):
        f.write("\n" + test_ids[i] + "," + str(predictions[i]))
    f.close()