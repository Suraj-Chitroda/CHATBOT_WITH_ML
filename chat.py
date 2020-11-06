import nltk
from nltk.stem.lancaster import LancasterStemmer
import numpy
import tflearn
import tensorflow
import random
import json
import pickle

stemmer = LancasterStemmer()

with open("json_file/intents.json") as file:
    data = json.load(file)

try:
    with open('data.pickle', "rb") as f:
        words, labels, training, output = pickle.load(f)
except:
    words = []
    labels = []
    docs_x = []
    docs_y = []

    for intent in data['intents']:
        for pattern in intent['patterns']:
            wrd = nltk.word_tokenize(pattern)
            words.extend(wrd)
            docs_x.append(wrd)
            docs_y.append(intent["tag"])

        labels.append(intent['tag'])

    words = [stemmer.stem(w.lower()) for w in words if w != '?']
    words = sorted(list(set(words)))
    labels = sorted(labels)

    training = []
    output = []

    empty_out = [0] * len(labels)

    for x, doc in enumerate(docs_x):
        bag = []
        wrd1 = [stemmer.stem(w.lower()) for w in doc]

        for w in words:
            if w in wrd1:
                bag.append(1)
            else:
                bag.append(0)

        output_row = empty_out[:]
        output_row[labels.index(docs_y[x])] = 1

        training.append(bag)
        output.append(output_row)

    with open('data.pickle', "wb") as f:
        pickle.dump((words, labels, training, output), f)

training = numpy.array(training)
output = numpy.array(output)

print("TRAINING {}".format(training))
print("OUTPUT {}".format(output))

tensorflow.reset_default_graph()

net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation='softmax')
net = tflearn.regression(net)

model = tflearn.DNN(net)

try:
    model.load('model.tflearn')
except expression as identifier:
    model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
    model.save('model.tflearn')


def bag_of_word(s, words):
    bag = [0] * len(words)

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1
    return numpy.array(bag)


def chat():
    print("Start chatting with CHATBOT! (type quit to stop)")

    while True:
        inp = input("type: ")
        if inp.lower() == 'quit':
            break
        results = model.predict([bag_of_word(inp, words)])[0]
        results_index = numpy.argmax(results)
        tag = labels[results_index]

        if results[results_index] > 0.7:
            for i in data['intents']:
                if i['tag'] == tag:
                    res = i['responses']
            print(random.choice(res))
        else:
            print("I don't get that...say it again")


if __name__ == "__main__":
    chat()
