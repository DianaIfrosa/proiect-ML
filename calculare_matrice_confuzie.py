if __name__ == '__main__':
    f = open(".\\predictions.txt")

    labels = []
    predictions = []

    line = f.readline()
    line = f.readline() # skip the first line that marks the columns

    while line != "":
        label = int(line[0])
        prediction = int(line[2])
        labels.append(label)
        predictions.append(prediction)
        line = f.readline()

    # build the matrix
    matrix = [[0,0,0], [0,0,0], [0,0,0]]
    for label, pred in zip(labels, predictions):
        matrix[label-1][pred-1] += 1

    print(matrix)

