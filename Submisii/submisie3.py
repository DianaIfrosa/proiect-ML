import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer

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

    create_vocab(np.concatenate((train_samples, validation_samples))) # find all distinct words in train and validation files
    f = open("./informations.txt", "w", encoding='utf8')
    f.write("There are " + str(len(vocab)) + " distinct words in train+validation\n")
    f.close()
    
    # try bag of words with a percentage of the  most frequent words
    features_percentage= [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
        
    # use model Naive Bayes with Multinomial distribution
    model = MultinomialNB() 
    max_accuracy = 0
    best_percentage = 0
        
    for percentage in features_percentage:
        vect = CountVectorizer(max_features=int(percentage*len(vocab)))
        train_data = vect.fit_transform(train_samples)
        validation_data = vect.transform(validation_samples)

        model.fit(train_data, train_labels)

        predictions = model.predict(validation_data)

        accuracy = np.mean(predictions == validation_labels)
        if accuracy > max_accuracy: # save the best parameter
            max_accuracy = accuracy
            best_percentage = percentage

        f = open("./accuracy-NB-vectorizer.txt", "a", encoding='utf8')
        f.write("Accuracy for " + str(percentage*100) + "% most frequent words: " + str(accuracy) + "\n")
        f.close()
        path = "./predictions-NB-vectorizer-" + str(percentage*100) + ".txt" 
        f = open(path, "w", encoding='utf8')
        f.write("Correct, wrong")
        for i in range(predictions.size):
            f.write("\n" + str(validation_labels[i]) + " " + str(predictions[i]))
        f.close()

    f = open("./informations.txt", "a", encoding='utf8')
    f.write("Best accuracy: " + str(max_accuracy) + ", best percentage: " + str(best_percentage) + "\n") 
    f.close()   
    
    vect = CountVectorizer(max_features=int(best_percentage*len(vocab)))    
    train_and_validation_data = vect.fit_transform(np.concatenate((train_samples, validation_samples)))
    model.fit(train_and_validation_data, np.concatenate((train_labels, validation_labels)))
    
    test_data = vect.transform(test_samples)
    predictions = model.predict(test_data)

    f = open("./test_labels_NB-vectorizer.txt", "w", encoding='utf8')
    f.write("id,label")
    for i in range(predictions.size):
        f.write("\n" + test_ids[i] + "," + str(predictions[i]))
    f.close()