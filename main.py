import numpy as np

from sklearn import svm, model_selection
from sklearn.model_selection import RepeatedStratifiedKFold, RepeatedKFold

def dbg():
    import pdb; pdb.set_trace()

train = np.loadtxt('features.train.txt')
test = np.loadtxt('features.test.txt')

X_train = train[:,1:] #started from the 1st index from the left and took them all.
Y_train = train[:,0]   #Just take 0st column
N_train = X_train[:, 0].size  #Just take 0st column's size

X_test = test[:,1:] #started from the 1st index from the left and took them all.
Y_test = test[:,0]  #Just take 0st column
N_test = X_test[:, 0].size #Just take 0st column's size

C = .01 #Here we assign Q and C values.C values is that 0<a n<C so it is  restricts  values of a.C parameter in SVM is Penalty parameter of the error term
Q = 2  #That is degree of polynomial


def make_binary(Ys, classToKeep):
    return np.array([1 if y == classToKeep else -1 for y in Ys])
#

def score(y_class):
    # Set all other labels to 0 to make it a binary y_class vs all classification
    bin_Y_train = make_binary(Y_train, y_class)
    bin_Y_test = make_binary(Y_test, y_class)

    clf = svm.SVC(kernel='poly', C=C, degree=Q, gamma=1.0, coef0=1.0, cache_size=20000)   #That is Svm Algorithm with polynomial
    clf.fit(X_train, bin_Y_train)
    print(f"Score of {y_class} versus all was {clf.score(X_test, bin_Y_test)}")


#That is first question and  I use library so That  was  implemented  easily .
for x in range(2,9, 2):
 score(x)

print("First Question and answer is E  that is max E in Error\n ")

for x in range(1,10, 2):
 score(x)

print("So the lowest   Ein was at a 1 versus all\n ")


def get_num_svs(y_class):
    # Set all other labels to 0 to make it a binary y_class vs all classification
    bin_Y_train = make_binary(Y_train, y_class)
    bin_Y_test = make_binary(Y_test, y_class)

    clf = svm.SVC(kernel='poly', C=C, degree=Q, gamma=1.0, coef0=1.0, cache_size=20000)
    clf.fit(X_train, bin_Y_train)
    return sum(clf.n_support_)

answer =get_num_svs(0) - get_num_svs(1);
print ("As u can see thats answer is :"+str(answer)+"so it is c\n")
#Here we assign Q and C values.C values is that 0<a n<C so it is  restricts  values of a.
Q = 2
Cs = [.001, .01, .1, 1]


def get_1v1_data(data, fst, snd):
    X, Y = data
    idxs = (Y == fst) | (Y == snd)
    return (X[idxs], Y[idxs])


def score_1v5(C):
    # Keep only data for 1s and 5s
    X_test1v5, Y_test1v5 = get_1v1_data((X_test, Y_test), 1, 5)
    X_train1v5, Y_train1v5 = get_1v1_data((X_train, Y_train), 1, 5)

    clf = svm.SVC(kernel='poly', C=C, degree=Q, gamma=1.0, coef0=1.0, cache_size=20000)
    clf.fit(X_train1v5, Y_train1v5)
    print(f"# support vectors for 1 vs 5 with C={C} was {sum(clf.n_support_)}")
    print(f"E_in for 1 vs 5 with C={C} was {clf.score(X_test1v5, Y_test1v5)}")
    print()


[score_1v5(C) for C in Cs]

X_test = test[:,1:]
Y_test = test[:,0]
N_test = X_test[:, 0].size

Qs = [2, 5]
Cs = [.0001, .001, .01, 1]


def score_1v5(C, Q):
    # Keep only data for 1s and 5s
    X_test1v5, Y_test1v5 = get_1v1_data((X_test, Y_test), 1, 5)
    X_train1v5, Y_train1v5 = get_1v1_data((X_train, Y_train), 1, 5)

    clf = svm.SVC(kernel='poly', C=C, degree=Q, gamma=1.0, coef0=1.0, cache_size=20000)
    clf.fit(X_train1v5, Y_train1v5)
    print(f"# support vectors with C={C} at Q={Q} was {sum(clf.n_support_)}")
    print(f"E_in with C={C} at Q={Q} was {clf.score(X_test1v5, Y_test1v5)}")
    print()


[score_1v5(C, Q) for C in Cs for Q in Qs]

Q = 2
RUNS = 100

X_test1v5,Y_test1v5 = get_1v1_data((X_test, Y_test), 1, 5)
X_train1v5, Y_train1v5 = get_1v1_data((X_train, Y_train), 1, 5)

Cs = [.0001, .001, .01, .1, 1]

def x_validate(C):
    clf = svm.SVC(kernel='poly', C=C, degree=Q, gamma=1.0, coef0=1.0, cache_size=20000)
    scores = model_selection.cross_val_score(clf, X_train1v5, Y_train1v5, cv=10)
    return scores.mean()

# We're being lazy and calculating the expected value of E_CV intead of selecting after each run, let's see if that works
E_cv_a = np.array([x_validate(Cs[0]) for _ in range(RUNS)]).mean()
print(f"E_CV for [a] is {E_cv_a}.")
E_cv_b = np.array([x_validate(Cs[1]) for _ in range(RUNS)]).mean()
print(f"E_CV for [b] is {E_cv_b}.")
E_cv_c = np.array([x_validate(Cs[2]) for _ in range(RUNS)]).mean()
print(f"E_CV for [c] is {E_cv_c}.")
E_cv_d = np.array([x_validate(Cs[3]) for _ in range(RUNS)]).mean()
print(f"E_CV for [d] is {E_cv_d}.")
E_cv_e = np.array([x_validate(Cs[4]) for _ in range(RUNS)]).mean()
print(f"E_CV for [e] is {E_cv_e}.")

rskf = RepeatedStratifiedKFold(n_splits=10, n_repeats=100)
winners = []
for train_idx, val_idx in rskf.split(X_train1v5, Y_train1v5):
    X_train, X_val = X_train1v5[train_idx], X_train1v5[val_idx]
    Y_train, Y_val = Y_train1v5[train_idx], Y_train1v5[val_idx]

    bestScore = 0
    winner = None
    for C in Cs:
        clf = svm.SVC(kernel='poly', C=C, degree=Q, gamma=1.0, coef0=1.0, cache_size=20000)
        clf.fit(X_train, Y_train)
        score = clf.score(X_val, Y_val)
        if score > bestScore:
            bestScore = score
            winner = C

    winners.append(winner)

Cs, counts = np.unique(winners, return_counts=True)
print("Cs: ", Cs)
print("Counts: ", counts)
print("\n")


C = 1e-3
rskf = RepeatedStratifiedKFold(n_splits=10, n_repeats=100)
clf = svm.SVC(kernel='poly', C=C, degree=Q, gamma=1.0, coef0=1.0, cache_size=20000)

scores = model_selection.cross_val_score(clf, X_train1v5, Y_train1v5, cv=rskf)
print("Questin 8 answer is ",1 - scores.mean())

Cs = [0.01, 1, 100, 1e4, 1e6]

X_test1v5, Y_test1v5 = get_1v1_data((X_test, Y_test), 1, 5)
X_train1v5, Y_train1v5 = get_1v1_data((X_train, Y_train), 1, 5)


def compute_eIN(C):
    clf = svm.SVC(kernel='rbf', C=C, degree=Q, gamma=1.0, cache_size=20000)
    clf.fit(X_train1v5, Y_train1v5)
    return 1 - clf.score(X_train1v5, Y_train1v5)


print([compute_eIN(C) for C in Cs])

Cs = [0.01, 1, 100, 1e4, 1e6]

X_test1v5, Y_test1v5 = get_1v1_data((X_test, Y_test), 1, 5)
X_train1v5, Y_train1v5 = get_1v1_data((X_train, Y_train), 1, 5)


def compute_eOUT(C):
    clf = svm.SVC(kernel='rbf', C=C, degree=Q, gamma=1.0, cache_size=20000)
    clf.fit(X_train1v5, Y_train1v5)
    return 1 - clf.score(X_test1v5, Y_test1v5)


print([compute_eOUT(C) for C in Cs])