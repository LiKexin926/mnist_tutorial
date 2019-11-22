from sklearn.datasets import fetch_mldata
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import LinearSVC

def log(X_train, X_test, Y_train, Y_test):
    log = LogisticRegression(multi_class='ovr',solver='liblinear')
    log.fit(X_train, Y_train)
    train_prediction = log.predict(X_train)
    test_prediction = log.predict(X_test)

    train_accuracy = metrics.accuracy_score(Y_train, train_prediction, normalize=True, sample_weight=None)
    test_accuracy = metrics.accuracy_score(Y_test, test_prediction, normalize=True, sample_weight=None)

    print('LogisticRegression_Training accuracy: %0.2f%%' % (train_accuracy*100))
    print('LogisticRegression_Testing accuracy: %0.2f%%' % (test_accuracy*100))

def bern(X_train, X_test, Y_train, Y_test):
    clf = BernoulliNB()  
    clf.fit(X_train, Y_train)  
    # BernoulliNB(alpha=1.0, binarize=0.0, class_prior=None, fit_prior=True)  
    train_prediction = clf.predict(X_train)
    test_prediction = clf.predict(X_test)

    train_accuracy = metrics.accuracy_score(Y_train, train_prediction, normalize=True, sample_weight=None)
    test_accuracy = metrics.accuracy_score(Y_test, test_prediction, normalize=True, sample_weight=None)

    print('BernoulliNB_Training accuracy: %0.2f%%' % (train_accuracy*100))
    print('BernoulliNB_Testing accuracy: %0.2f%%' % (test_accuracy*100))

def svc(X_train, X_test, Y_train, Y_test):
    svc = LinearSVC(max_iter = 10000)
    svc.fit(X_train, Y_train)
    train_prediction = svc.predict(X_train)
    test_prediction = svc.predict(X_test)

    train_accuracy = metrics.accuracy_score(Y_train, train_prediction, normalize=True, sample_weight=None)
    test_accuracy = metrics.accuracy_score(Y_test, test_prediction, normalize=True, sample_weight=None)

    print('LinearSVC_Training accuracy: %0.2f%%' % (train_accuracy*100))
    print('LinearSVC_Testing accuracy: %0.2f%%' % (test_accuracy*100))

def svm(X_train, X_test, Y_train, Y_test):
    svc = LinearSVC(loss = 'hinge',tol = 1e-3,C = 0.1,max_iter = 10000)
    svc.fit(X_train, Y_train)
    train_prediction = svc.predict(X_train)
    test_prediction = svc.predict(X_test)

    train_accuracy = metrics.accuracy_score(Y_train, train_prediction, normalize=True, sample_weight=None)
    test_accuracy = metrics.accuracy_score(Y_test, test_prediction, normalize=True, sample_weight=None)

    print('adjust_parameters_Training accuracy: %0.2f%%' % (train_accuracy*100))
    print('adjust_parameters_Testing accuracy: %0.2f%%' % (test_accuracy*100))


if __name__ == "__main__":
    # download and read mnist
    mnist = fetch_mldata('MNIST original')

    X = mnist.data / 255.
    Y = mnist.target
    X_train, X_test, Y_train, Y_test = train_test_split(X[::10], Y[::10], test_size=1000)

    log(X_train, X_test, Y_train, Y_test)
    bern(X_train, X_test, Y_train, Y_test)
    svc(X_train, X_test, Y_train, Y_test)
    svm(X_train, X_test, Y_train, Y_test)