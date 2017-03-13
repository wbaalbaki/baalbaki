from collections import defaultdict

hist = defaultdict(int)

questions = open('data/squad/train.question', 'rt')



for question in questions:
    prevQuestion = question.split()
    question = prevQuestion[0]
    if question=="In":
        question = question + ' ' + prevQuestion[1]
    hist[question] += 1

import operator
hist_x = sorted(hist.items(), key=operator.itemgetter(1))


a = [el[0] for el in hist_x[len(hist_x)-7:]]
b = [el[1] for el in hist_x[len(hist_x)-7:]]
print(a, b)

import matplotlib.pyplot as plt
N = len(b)
x = range(N)
width = 1/1.5
plt.bar(x, b, width, color="blue")

toPlot = [x_i + 0.35 for x_i in x]
plt.xticks(toPlot, a)

plt.show()