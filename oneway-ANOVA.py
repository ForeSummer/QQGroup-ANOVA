#! python2
# coding: utf-8

import pandas as pd
import numpy as np
import scipy.stats as stats
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import seaborn as sns
import svm
import rf

df = pd.read_csv('data.csv')


def gaussian(x, *param):
    return param[0] * np.exp(-np.power(x - param[1], 2.) / (2 * np.power(param[2], 2.)))


def fittingGaussian(colName):
    data = df[colName].round().value_counts().sort_index()
    x = data.index.values
    y = data.values
    popt, pcov = curve_fit(gaussian, x, y, p0=[0, 1, 2])

    sns.distplot(df[colName], label='Col[7]')
    plt.title('Fitting Gaussian')
    plt.show()


def grouping(colName, mode=False):
    arr = []

    for i in range(1, 6):
        data = df[df['group_category'] == i][colName].values
        if mode:
            arr.append(stats.boxcox(data)[0])
        else:
            arr.append(data)

    return arr


def oneway(arr, name):
    f, p = stats.f_oneway(*arr)
    print '%s: one-way ANOVA(f=%s,p=%s).' % (name, f, p)


def normalTest(data, name):
    pval = stats.normaltest(data)[1]
    if (pval < 0.05):
        print "%s: are not normal distributed(p=%s)." % (name, pval)
    else:
        print "%s: are normal distributed(p=%s)." % (name, pval)


def leveneTest(arr, name):
    pval = stats.levene(*arr)[1]
    if (pval < 0.05):
        print "%s: are not the homogeneity of variances(p=%s)." % (name, pval)
    else:
        print "%s: are the homogeneity of variances(p=%s)." % (name, pval)


def result():
    sns.set(palette="muted", color_codes=True)
    # 3a
    print '============================================='
    print 'HW1-3a'
    print '============================================='
    # fittingGaussian('average_age')
    normalTest(df['average_age'].values, name='Col[7]')

    # 3b
    print '============================================='
    print 'HW1-3b'
    print '============================================='
    arr_7 = grouping('average_age')
    for i in range(0, 5):
        normalTest(arr_7[i], name='Col[7] in Group %s' % (i + 1))

    print '---------------------------------------------'
    leveneTest(arr_7, name='Col[7]')

    # 3c
    print '============================================='
    print 'HW1-3c'
    print '============================================='
    oneway(arr_7, name='Col[7]')

    # 4
    print '============================================='
    print 'HW1-4'
    print '============================================='
    selectCol = ['message_number', 'variance_age', 'conversation_number']

    for key in selectCol:
        normalTest(df[key].values, name='Col[%s]' % key)

    print '---------------------------------------------'

    for key in selectCol:
        normalTest(np.log(df[key].values), name='log Col[%s]' % key)

    # 5b
    print '============================================='
    print 'HW1-5'
    print '============================================='

    # no box-cox
    for key in selectCol:
        arr = grouping(key)
        oneway(arr, 'Col[%s]' % key)

    print '---------------------------------------------'

    # box-cox
    for key in selectCol:
        arr = grouping(key, mode=True)
        oneway(arr, 'box-cox Col[%s]' % key)

    # 6
    print '============================================='
    print 'HW1-5'
    print '============================================='
    testSizeArr = [0.6, 0.5, 0.4]

    print "SVM ['average_age', 'variance_age']"
    for size in testSizeArr:
        train_score, test_score = svm.run(
            df[['average_age', 'variance_age']], df['group_category'], size)

        print '[train:test = %s:%s] train correct=%s,test correct=%s.' % (1 - size, size, train_score, test_score)

    print '---------------------------------------------'

    print "Random Forest ['average_age', 'variance_age']"
    for size in testSizeArr:
        train_score, test_score = rf.run(
            df[['average_age', 'variance_age']], df['group_category'], size)

        print '[train:test = %s:%s] train correct=%s,test correct=%s.' % (1 - size, size, train_score, test_score)

def draw():
    sns.set(palette="muted", color_codes=True)

    # 3a
    sns.distplot(df['average_age'], kde_kws={"label": "Col[7]"})
    plt.title('Fitting Gaussian')
    plt.savefig('3a.png')
    plt.close()

    # 3b
    arr_7 = grouping('average_age')
    for i in range(1, 6):
        sns.distplot(df[df['group_category'] == i]['average_age'], kde_kws={"label": "Group %s" % i})

    plt.title('Fitting Gaussian')
    plt.savefig('3b.png')
    plt.close()

    # 4
    selectCol = ['message_number', 'variance_age', 'conversation_number']
    for key in selectCol:
        sns.distplot(df[key], kde_kws={"label": key})
        plt.title('Fitting Gaussian')
        plt.savefig('4-%s.png' % key)
        plt.close()
        sns.distplot(np.log(df[key]), kde_kws={"label": "log-%s" % key})
        plt.title('Fitting Gaussian')
        plt.savefig('4-log-%s.png' % key)
        plt.close()




if __name__ == "__main__":
    # result()
    draw()