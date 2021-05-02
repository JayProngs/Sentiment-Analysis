from flask import Flask, flash, redirect, render_template, request, session, abort
import os
import csv
import math
import random
import pickle
from textblob import TextBlob
from textblob.classifiers import NaiveBayesClassifier

a = []
b = []
sec = {'jayesh': 'prongs', 'vedant': 'vedant', 'kapil': 'kapil'}
app = Flask(__name__)


@app.route('/')
def main():
    return render_template('home.html')


@app.route('/homepage')
def homepage():
    return render_template('home.html')


@app.route('/home')
def home():
    if not session.get('logged_in'):
        return render_template('login.html')
    else:
        return render_template('comment.html')


@app.route('/signup')
def signup():
    return render_template('signin.html')


@app.route('/signin', methods=['POST'])
def do_signup():
    user = request.form['form-username']
    pswd = request.form['form-password']
    sec.update({user:pswd})
    session['logged_in'] = True
    return home()


@app.route('/login', methods=['POST'])
def do_admin_login():
    b = sec.keys()
    for i in b:
        if i == request.form['form-username']:
            if request.form['form-password'] == sec.get(i):
                session['logged_in'] = True
    if not session.get('logged_in'):
        error='Invalid Username or password. Please Try again!'
        return render_template('login.html',error=error)
    else:
        return render_template('comment.html')


@app.route('/result', methods=['POST'])
def result():
    data = request.form['comment']
    sent = TextBlob(data)
    num = sent.sentiment.polarity
    a.clear()
    if num==0:
        a.insert(0, "Neutral")
        your_list= [tuple(row) for row in csv.reader(open('data.csv', 'r'))]
    if num > 0:
        a.insert(0, "Positive")
        your_list= [tuple(row) for row in csv.reader(open('positive.csv', 'r'))]
    if num < 0:
        a.insert(0, "Negative")
        your_list= [tuple(row) for row in csv.reader(open('negative.csv', 'r'))]
    cl = NaiveBayesClassifier(your_list)
    blob = TextBlob(data, classifier=cl)
    emo = blob.classify()
    a.insert(1, emo)
    if emo == "Anger":
        return render_template('anger.html', a=a, data=data)
    elif emo == "Anticipation":
        return render_template('anticipation.html', a=a, data=data)
    elif emo == "Disgust":
        a.insert(0, "Negative")
        return render_template('disgust.html', a=a, data=data)
    elif emo == "Fear":
        return render_template('fear.html', a=a, data=data)
    elif emo == "Joy":
        return render_template('joy.html', a=a, data=data)
    elif emo == "Sadness":
        return render_template('sadness.html', a=a, data=data)
    elif emo == "Surprise":
        a.insert(0, "Positive")
        return render_template('surprise.html', a=a, data=data)
    elif emo == "Trust":
        return render_template('trust.html', a=a, data=data)
    else:
        return render_template('result.html', a=a, data=data)


@app.route("/logout")
def logout():
    session['logged_in'] = False
    return homepage()


def loadCsv(filename):
    lines = csv.reader(open(r'C:\Users\jayes\Desktop\Expresso\dataset1.csv'))
    set = list(lines)
    for i in range(len(set)):
        set[i] = [float(x) for x in set[i]]
    return set


def splitDataset(set, splitRatio):
    trainSize = int(len(set) * splitRatio)
    trainSet = []
    copy = list(set)
    while len(trainSet) < trainSize:
        index = random.randrange(len(copy))
        trainSet.append(copy.pop(index))
    return [trainSet, copy]


def separateByClass(set):
    separated = {}
    for i in range(len(set)):
        vector = set[i]
        if (vector[-1] not in separated):
            separated[vector[-1]] = []
        separated[vector[-1]].append(vector)
    return separated


def mean(numbers):
    return sum(numbers) / float(len(numbers))


def stdev(numbers):
    avg = mean(numbers)
    variance = sum([pow(x - avg, 2) for x in numbers]) / float(len(numbers) - 1)
    return math.sqrt(variance)


def summarize(set):
    summaries = [(mean(attribute), stdev(attribute)) for attribute in zip(*set)]
    del summaries[-1]
    return summaries


def summarizeByClass(dataset):
    separated = separateByClass(dataset)
    summaries = {}
    for classValue, instances in separated.items():
        summaries[classValue] = summarize(instances)
    return summaries


def calculateProbability(x, mean, stdev):
    exponent = math.exp(-(math.pow(x - mean, 2) / (2 * math.pow(stdev, 2))))
    return (1 / (math.sqrt(2 * math.pi) * stdev)) * exponent


def calculateClassProbabilities(summaries, inputVector):
    probabilities = {}
    for classValue, classSummaries in summaries.items():
        probabilities[classValue] = 1
        for i in range(len(classSummaries)):
            mean, stdev = classSummaries[i]
            x = inputVector[i]
            probabilities[classValue] *= calculateProbability(x, mean, stdev)
    return probabilities


def predict(summaries, inputVector):
    probabilities = calculateClassProbabilities(summaries, inputVector)
    bestLabel, bestProb = None, -1
    for classValue, probability in probabilities.items():
        if bestLabel is None or probability > bestProb:
            bestProb = probability
            bestLabel = classValue
    return bestLabel


def getPredictions(summaries, testSet):
    predictions = []
    for i in range(len(testSet)):
        result = predict(summaries, testSet[i])
        predictions.append(result)
    return predictions


def getAccuracy(testSet, predictions):
    correct = 0
    for x in range(len(testSet)):
        if testSet[x][-1] == predictions[x]:
            correct += 1
    return (correct / float(len(testSet))) * 100.0


def senti():
    filename = 'dataset1.csv'
    splitRatio = 0.75
    set = loadCsv(filename)
    trainingSet, testSet = splitDataset(set, splitRatio)
    print('Split {0} rows into train = {1} and test = {2} rows'.format(len(set), len(trainingSet), len(testSet)))
    # prepare model
    summaries = summarizeByClass(trainingSet)
    # test model
    predictions = getPredictions(summaries, testSet)
    accuracy = getAccuracy(testSet, predictions)
    print('Accuracy: {0}%'.format(accuracy))


if __name__ == "__main__":
    app.secret_key = os.urandom(12)
app.run(debug=True)
