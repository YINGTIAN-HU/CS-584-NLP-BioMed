import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from nltk import word_tokenize, sent_tokenize
from nltk.stem import *
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb

def load_data_as_data_frame(f_path):
    '''
        Given a path, loads a data set and puts it into a dataframe
        - simplified mechanism
    '''
    df = pd.read_csv(f_path)
    return df

def preprocess_text(raw_text):
    '''
        Preprocessing function
        PROGRAMMING TIP: Always a good idea to have a *master* preprocessing function that reads in a string and returns the
        preprocessed string after applying a series of functions.
    '''
    # stemming and lowercasing (no stopword removal
    words = [stemmer.stem(w) for w in raw_text.lower().split()]
    return (" ".join(words))

def GNBGridSearch(params, skf, train_data, train_label, train_engineer, feature_set):
    K = skf.get_n_splits(train_data, train_label)
    accuracy = np.zeros(K)
    f1_micro = np.zeros(K)
    f1_macro = np.zeros(K)
    oof_prediction = np.zeros(len(train_label))
    f1_micro_max = 0
    for C in params['C']:
        k = 0
        for train_index, test_index in skf.split(train_data, train_label):

            train_data_train = map(train_data.__getitem__, train_index)
            train_data_val = map(train_data.__getitem__, test_index)
            train_engineer_train = train_engineer.iloc[train_index].reset_index()
            train_engineer_val = train_engineer.iloc[test_index].reset_index()

            train_label_train = train_label[train_index]
            train_label_val = train_label[test_index]

            # Vectorization
            vectorizer = CountVectorizer(ngram_range=(1, 3), max_features=10000, analyzer='word', stop_words='english')
            train_data_train_vectors = vectorizer.fit_transform(train_data_train).toarray()
            train_data_val_vectors = vectorizer.transform(train_data_val).toarray()
            train_data_train_vectors = pd.DataFrame(train_data_train_vectors, columns=vectorizer.get_feature_names())
            train_data_val_vectors = pd.DataFrame(train_data_val_vectors, columns=vectorizer.get_feature_names())

            # combine engineered feature
            train_data_train_vectors = pd.concat([train_data_train_vectors, train_engineer_train], axis=1)
            train_data_val_vectors = pd.concat([train_data_val_vectors, train_engineer_val], axis=1)

            # get subsect
            if feature_set is not None:
                train_data_train_vectors = train_data_train_vectors[feature_set]
                train_data_val_vectors = train_data_val_vectors[feature_set]
            # Train the model

            # Train the model
            gnb = GaussianNB().fit(train_data_train_vectors, train_label_train)

            # Prediction
            oof_prediction[test_index] = gnb.predict(train_data_val_vectors)
            accuracy[k] = accuracy_score(oof_prediction[test_index], train_label_val)
            f1_micro[k] = f1_score(oof_prediction[test_index], train_label_val, average='micro')
            f1_macro[k] = f1_score(oof_prediction[test_index], train_label_val, average='macro')
            # print(confusion_matrix(train_label_val, oof_prediction[test_index]))
            # print(train_label_val)
            # print(oof_prediction[test_index])
            k += 1

            # print results
        # print('C ' + str(C))
        print('accuracy ' + "{:.3f}".format(np.mean(accuracy)))
        print('f1_micro ' + "{:.3f}".format(np.mean(f1_micro)))
        print('f1_macro ' + "{:.3f}".format(np.mean(f1_macro)))
        if np.mean(f1_micro) > f1_micro_max:
            f1_micro_max = np.mean(f1_micro)
            best_param = {"C": 0}
            best_metric = [np.mean(accuracy), np.mean(f1_micro), np.mean(f1_macro)]
    return best_param, best_metric

def LogisticRegressionGridSearch(params, skf, train_data, train_label, train_engineer, feature_set):
    K = skf.get_n_splits(train_data, train_label)
    accuracy = np.zeros(K)
    f1_micro = np.zeros(K)
    f1_macro = np.zeros(K)
    oof_prediction = np.zeros(len(train_label))
    f1_micro_max = 0
    for C in params['C']:
        k = 0
        for train_index, test_index in skf.split(train_data, train_label):

            train_data_train = map(train_data.__getitem__, train_index)
            train_data_val = map(train_data.__getitem__, test_index)
            train_engineer_train = train_engineer.iloc[train_index].reset_index()
            train_engineer_val = train_engineer.iloc[test_index].reset_index()

            train_label_train = train_label[train_index]
            train_label_val = train_label[test_index]

            # Vectorization
            vectorizer = CountVectorizer(ngram_range=(1, 3), max_features=10000, analyzer='word', stop_words='english')
            train_data_train_vectors = vectorizer.fit_transform(train_data_train).toarray()
            train_data_val_vectors = vectorizer.transform(train_data_val).toarray()
            train_data_train_vectors = pd.DataFrame(train_data_train_vectors, columns=vectorizer.get_feature_names())
            train_data_val_vectors = pd.DataFrame(train_data_val_vectors, columns=vectorizer.get_feature_names())

            # combine engineered feature
            train_data_train_vectors = pd.concat([train_data_train_vectors, train_engineer_train], axis=1)
            train_data_val_vectors = pd.concat([train_data_val_vectors, train_engineer_val], axis=1)

            # get subsect
            if feature_set is not None:
                train_data_train_vectors = train_data_train_vectors[feature_set]
                train_data_val_vectors = train_data_val_vectors[feature_set]
            # Train the model

            # Train the model
            lr = LogisticRegression(C=C, random_state=0, n_jobs=-1, solver='lbfgs').fit(train_data_train_vectors,
                                                                                        train_label_train)

            # Prediction
            oof_prediction[test_index] = lr.predict(train_data_val_vectors)
            accuracy[k] = accuracy_score(oof_prediction[test_index], train_label_val)
            f1_micro[k] = f1_score(oof_prediction[test_index], train_label_val, average='micro')
            f1_macro[k] = f1_score(oof_prediction[test_index], train_label_val, average='macro')
            # print(confusion_matrix(train_label_val, oof_prediction[test_index]))
            # print(train_label_val)
            # print(oof_prediction[test_index])
            k += 1

            # print results
        print('C ' + str(C))
        print('accuracy ' + "{:.3f}".format(np.mean(accuracy)))
        print('f1_micro ' + "{:.3f}".format(np.mean(f1_micro)))
        print('f1_macro ' + "{:.3f}".format(np.mean(f1_macro)))
        if np.mean(f1_micro) > f1_micro_max:
            f1_micro_max = np.mean(f1_micro)
            best_param = {"C": C}
            best_metric = [np.mean(accuracy), np.mean(f1_micro), np.mean(f1_macro)]
    return best_param, best_metric

def SVMGridSearch(params, skf, train_data, train_label, train_engineer, feature_set):
    K = skf.get_n_splits(train_data, train_label)
    accuracy = np.zeros(K)
    f1_micro = np.zeros(K)
    f1_macro = np.zeros(K)
    oof_prediction = np.zeros(len(train_label))
    f1_micro_max = 0
    for C in params['C']:
        for kernel in params['kernel']:
            k = 0
            for train_index, test_index in skf.split(train_data, train_label):

                train_data_train = map(train_data.__getitem__, train_index)
                train_data_val = map(train_data.__getitem__, test_index)
                train_engineer_train = train_engineer.iloc[train_index].reset_index()
                train_engineer_val = train_engineer.iloc[test_index].reset_index()

                train_label_train = train_label[train_index]
                train_label_val = train_label[test_index]

                # Vectorization
                vectorizer = CountVectorizer(ngram_range=(1, 3), max_features=10000, analyzer='word',
                                             stop_words='english')
                train_data_train_vectors = vectorizer.fit_transform(train_data_train).toarray()
                train_data_val_vectors = vectorizer.transform(train_data_val).toarray()
                train_data_train_vectors = pd.DataFrame(train_data_train_vectors,
                                                        columns=vectorizer.get_feature_names())
                train_data_val_vectors = pd.DataFrame(train_data_val_vectors, columns=vectorizer.get_feature_names())

                # combine engineered feature
                train_data_train_vectors = pd.concat([train_data_train_vectors, train_engineer_train], axis=1)
                train_data_val_vectors = pd.concat([train_data_val_vectors, train_engineer_val], axis=1)

                # get subsect
                if feature_set is not None:
                    train_data_train_vectors = train_data_train_vectors[feature_set]
                    train_data_val_vectors = train_data_val_vectors[feature_set]
                # Train the model

                # Train the model
                SVM = svm.SVC(C=C, gamma='scale', kernel=kernel).fit(train_data_train_vectors, train_label_train)

                # Prediction
                oof_prediction[test_index] = SVM.predict(train_data_val_vectors)
                accuracy[k] = accuracy_score(oof_prediction[test_index], train_label_val)
                f1_micro[k] = f1_score(oof_prediction[test_index], train_label_val, average='micro')
                f1_macro[k] = f1_score(oof_prediction[test_index], train_label_val, average='macro')
                # print(confusion_matrix(train_label_val, oof_prediction[test_index]))
                # print(train_label_val)
                # print(oof_prediction[test_index])
                k += 1

                # print results
            # print('C ' + str(C))
            print('accuracy ' + "{:.3f}".format(np.mean(accuracy)))
            print('f1_micro ' + "{:.3f}".format(np.mean(f1_micro)))
            print('f1_macro ' + "{:.3f}".format(np.mean(f1_macro)))
            if np.mean(f1_micro) > f1_micro_max:
                f1_micro_max = np.mean(f1_micro)
                best_param = {"C": C, 'kernel': kernel}
                best_metric = [np.mean(accuracy), np.mean(f1_micro), np.mean(f1_macro)]
    return best_param, best_metric

def RandomForestGridSearch(params, skf, train_data, train_label, train_engineer, feature_set):
    K = skf.get_n_splits(train_data, train_label)
    accuracy = np.zeros(K)
    f1_micro = np.zeros(K)
    f1_macro = np.zeros(K)
    oof_prediction = np.zeros(len(train_label))
    f1_micro_max = 0
    for n_estimators in params['n_estimators']:
        for max_depth in params['max_depth']:
            for min_samples_split in params['min_samples_split']:
                for min_samples_leaf in params['min_samples_leaf']:
                    k = 0
                    for train_index, test_index in skf.split(train_data, train_label):

                        train_data_train = map(train_data.__getitem__, train_index)
                        train_data_val = map(train_data.__getitem__, test_index)
                        train_engineer_train = train_engineer.iloc[train_index].reset_index()
                        train_engineer_val = train_engineer.iloc[test_index].reset_index()

                        train_label_train = train_label[train_index]
                        train_label_val = train_label[test_index]

                        # Vectorization
                        vectorizer = CountVectorizer(ngram_range=(1, 3), max_features=10000, analyzer='word',
                                                     stop_words='english')
                        train_data_train_vectors = vectorizer.fit_transform(train_data_train).toarray()
                        train_data_val_vectors = vectorizer.transform(train_data_val).toarray()
                        train_data_train_vectors = pd.DataFrame(train_data_train_vectors,
                                                                columns=vectorizer.get_feature_names())
                        train_data_val_vectors = pd.DataFrame(train_data_val_vectors,
                                                              columns=vectorizer.get_feature_names())

                        # combine engineered feature
                        train_data_train_vectors = pd.concat([train_data_train_vectors, train_engineer_train], axis=1)
                        train_data_val_vectors = pd.concat([train_data_val_vectors, train_engineer_val], axis=1)

                        # get subsect
                        if feature_set is not None:
                            train_data_train_vectors = train_data_train_vectors[feature_set]
                            train_data_val_vectors = train_data_val_vectors[feature_set]
                        # Train the model

                        # Train the model
                        rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                                    min_samples_split=min_samples_split,
                                                    min_samples_leaf=min_samples_leaf,
                                                    random_state=0, n_jobs=-1)
                        rf = rf.fit(train_data_train_vectors, train_label_train)

                        # Prediction
                        oof_prediction[test_index] = rf.predict(train_data_val_vectors)
                        accuracy[k] = accuracy_score(oof_prediction[test_index], train_label_val)
                        f1_micro[k] = f1_score(oof_prediction[test_index], train_label_val, average='micro')
                        f1_macro[k] = f1_score(oof_prediction[test_index], train_label_val, average='macro')
                        # print(confusion_matrix(train_label_val, oof_prediction[test_index]))
                        # print(train_label_val)
                        # print(oof_prediction[test_index])
                        k += 1

                    # print results
                    print('n_estimators ' + str(n_estimators) + " max_depth " + str(max_depth) +
                          ' min_samples_split ' + str(min_samples_split) + ' min_samples_leaf ' + str(min_samples_leaf))
                    print('accuracy ' + "{:.3f}".format(np.mean(accuracy)))
                    print('f1_micro ' + "{:.3f}".format(np.mean(f1_micro)))
                    print('f1_macro ' + "{:.3f}".format(np.mean(f1_macro)))
                    if np.mean(f1_micro) > f1_micro_max:
                        f1_micro_max = np.mean(f1_micro)
                        best_param = {'n_estimators': n_estimators, 'max_depth': max_depth,
                                      'min_samples_split': min_samples_split, 'min_samples_leaf': min_samples_leaf}
                        best_metric = [np.mean(accuracy), np.mean(f1_micro), np.mean(f1_macro)]
    return best_param, best_metric

def KNNGridSearch(params, skf, train_data, train_label, train_engineer, feature_set):
    K = skf.get_n_splits(train_data, train_label)
    accuracy = np.zeros(K)
    f1_micro = np.zeros(K)
    f1_macro = np.zeros(K)
    oof_prediction = np.zeros(len(train_label))
    f1_micro_max = 0
    for n in params['n']:
        k = 0
        for train_index, test_index in skf.split(train_data, train_label):

            train_data_train = map(train_data.__getitem__, train_index)
            train_data_val = map(train_data.__getitem__, test_index)
            train_engineer_train = train_engineer.iloc[train_index].reset_index()
            train_engineer_val = train_engineer.iloc[test_index].reset_index()

            train_label_train = train_label[train_index]
            train_label_val = train_label[test_index]

            # Vectorization
            vectorizer = CountVectorizer(ngram_range=(1, 3), max_features=10000, analyzer='word', stop_words='english')
            train_data_train_vectors = vectorizer.fit_transform(train_data_train).toarray()
            train_data_val_vectors = vectorizer.transform(train_data_val).toarray()
            train_data_train_vectors = pd.DataFrame(train_data_train_vectors, columns=vectorizer.get_feature_names())
            train_data_val_vectors = pd.DataFrame(train_data_val_vectors, columns=vectorizer.get_feature_names())

            # combine engineered feature
            train_data_train_vectors = pd.concat([train_data_train_vectors, train_engineer_train], axis=1)
            train_data_val_vectors = pd.concat([train_data_val_vectors, train_engineer_val], axis=1)

            # get subsect
            if feature_set is not None:
                train_data_train_vectors = train_data_train_vectors[feature_set]
                train_data_val_vectors = train_data_val_vectors[feature_set]
            # Train the model
            knn = KNeighborsClassifier(n_neighbors=n, n_jobs=-1).fit(train_data_train_vectors, train_label_train)

            # Prediction
            oof_prediction[test_index] = knn.predict(train_data_val_vectors)
            accuracy[k] = accuracy_score(oof_prediction[test_index], train_label_val)
            f1_micro[k] = f1_score(oof_prediction[test_index], train_label_val, average='micro')
            f1_macro[k] = f1_score(oof_prediction[test_index], train_label_val, average='macro')
            # print(confusion_matrix(train_label_val, oof_prediction[test_index]))
            # print(train_label_val)
            # print(oof_prediction[test_index])
            k += 1

            # print results
        # print('C ' + str(C))
        print('accuracy ' + "{:.3f}".format(np.mean(accuracy)) + " std:" + "{:.3f}".format(np.std(accuracy)))
        print('f1_micro ' + "{:.3f}".format(np.mean(f1_micro)) + " std:" + "{:.3f}".format(np.std(f1_micro)))
        print('f1_macro ' + "{:.3f}".format(np.mean(f1_macro)) + " std:" + "{:.3f}".format(np.std(f1_macro)))
        if np.mean(f1_micro) > f1_micro_max:
            f1_micro_max = np.mean(f1_micro)
            best_param = {"n": n}
            best_metric = [np.mean(accuracy), np.mean(f1_micro), np.mean(f1_macro)]
    return best_param, best_metric

def XGBGridSearch(params, skf, train_data, train_label, train_engineer, feature_set):
    K = skf.get_n_splits(train_data, train_label)
    f1_micro_max = 0
    accuracy = np.zeros(K)
    f1_micro = np.zeros(K)
    f1_macro = np.zeros(K)
    oof_prediction = np.zeros(len(train_label))
    for n_estimators in params['n_estimators']:
        for max_depth in params['max_depth']:
            for min_child_weight in params['min_child_weight']:
                for gamma in params['gamma']:
                    for subsample in params['subsample']:
                        for colsample_bytree in params['colsample_bytree']:
                            for reg_alpha in params['reg_alpha']:
                                for reg_lambda in params['reg_lambda']:
                                    for learning_rate in params['learning_rate']:
                                        k = 0
                                        for train_index, test_index in skf.split(train_data, train_label):

                                            train_data_train = map(train_data.__getitem__, train_index)
                                            train_data_val = map(train_data.__getitem__, test_index)
                                            train_engineer_train = train_engineer.iloc[train_index].reset_index()
                                            train_engineer_val = train_engineer.iloc[test_index].reset_index()

                                            train_label_train = train_label[train_index]
                                            train_label_val = train_label[test_index]

                                            # Vectorization
                                            vectorizer = CountVectorizer(ngram_range=(1, 3), max_features=10000,
                                                                         analyzer='word', stop_words='english')
                                            train_data_train_vectors = vectorizer.fit_transform(
                                                train_data_train).toarray()
                                            train_data_val_vectors = vectorizer.transform(train_data_val).toarray()
                                            train_data_train_vectors = pd.DataFrame(train_data_train_vectors,
                                                                                    columns=vectorizer.get_feature_names())
                                            train_data_val_vectors = pd.DataFrame(train_data_val_vectors,
                                                                                  columns=vectorizer.get_feature_names())

                                            # print(train_data_train_vectors.shape)
                                            # print(train_data_val_vectors.shape)

                                            # print(train_engineer_train.shape)
                                            # print(train_engineer_val.shape)

                                            # combine engineered feature
                                            train_data_train_vectors = pd.concat(
                                                [train_data_train_vectors, train_engineer_train], axis=1)
                                            train_data_val_vectors = pd.concat(
                                                [train_data_val_vectors, train_engineer_val], axis=1)

                                            # get subsect
                                            if feature_set is not None:
                                                train_data_train_vectors = train_data_train_vectors[feature_set]
                                                train_data_val_vectors = train_data_val_vectors[feature_set]

                                            # print(train_data_train_vectors.shape)
                                            # print(train_data_val_vectors.shape)
                                            # print(len(test_index))
                                            # Train the model
                                            xgb_clf = xgb.XGBClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                                                        min_child_weight=min_child_weight, gamma=gamma,
                                                                        subsample=subsample,
                                                                        colsample_bytree=colsample_bytree,
                                                                        reg_alpha=reg_alpha, reg_lambda=reg_lambda,
                                                                        learning_rate=learning_rate, random_state=0,
                                                                        n_jobs=-1).fit(train_data_train_vectors,
                                                                                       train_label_train)  # https://xgboost.readthedocs.io/en/latest/tutorials/param_tuning.html

                                            # Prediction
                                            oof_prediction[test_index] = xgb_clf.predict(train_data_val_vectors)
                                            accuracy[k] = accuracy_score(oof_prediction[test_index], train_label_val)
                                            f1_micro[k] = f1_score(oof_prediction[test_index], train_label_val,
                                                                   average='micro')
                                            f1_macro[k] = f1_score(oof_prediction[test_index], train_label_val,
                                                                   average='macro')
                                            # print(confusion_matrix(train_label_val, oof_prediction[test_index]))
                                            # print(train_label_val)
                                            # print(oof_prediction[test_index])
                                            k += 1

                                        # print results
                                        print('n_estimators', n_estimators, 'max_depth', max_depth, 'min_child_weight',
                                              min_child_weight,
                                              'gamma', gamma, 'subsample', subsample, 'colsample_bytree',
                                              colsample_bytree, 'reg_alpha', reg_alpha,
                                              'reg_lambda', reg_lambda, 'learning_rate', learning_rate)
                                        print('accuracy ' + "{:.3f}".format(
                                            np.mean(accuracy)) + " std:" + "{:.3f}".format(np.std(accuracy)))
                                        print('f1_micro ' + "{:.3f}".format(
                                            np.mean(f1_micro)) + " std:" + "{:.3f}".format(np.std(f1_micro)))
                                        print('f1_macro ' + "{:.3f}".format(
                                            np.mean(f1_macro)) + " std:" + "{:.3f}".format(np.std(f1_macro)))
                                        if np.mean(f1_micro) > f1_micro_max:
                                            f1_micro_max = np.mean(f1_micro)
                                            best_param = {"n_estimators": n_estimators, "max_depth": max_depth,
                                                          "min_child_weight": min_child_weight, "gamma": gamma,
                                                          "subsample": subsample, "colsample_bytree": colsample_bytree,
                                                          "reg_alpha": reg_alpha, "reg_lambda": reg_lambda,
                                                          "learning_rate": learning_rate}

                                            best_metric = [np.mean(accuracy), np.mean(f1_micro), np.mean(f1_macro)]
    return best_param, best_metric

def ModelFit_Engineer_Feature(X_train, y_train, X_test, y_test, params, model, vectorizer, X_train_engineered, X_test_engineered, feature_set, p=1):
    # vectorizer = CountVectorizer(ngram_range=(1, 3), analyzer = 'word', tokenizer = None, preprocessor = None, max_features = 10000, stop_words= 'english')
    X_train_vectors = vectorizer.fit_transform(X_train).toarray()
    X_test_vectors = vectorizer.transform(X_test).toarray()
    X_train_vectors = pd.DataFrame(X_train_vectors, columns=vectorizer.get_feature_names())
    X_test_vectors = pd.DataFrame(X_test_vectors, columns=vectorizer.get_feature_names())

    X_train_vectors = pd.concat([X_train_vectors, X_train_engineered], axis=1)
    X_test_vectors = pd.concat([X_test_vectors, X_test_engineered], axis=1)

    # get subsect
    if feature_set is not None:
        X_train_vectors = X_train_vectors[feature_set]
        X_test_vectors = X_test_vectors[feature_set]
    # Train the model
    sub_training_set_size = int(p * X_train_vectors.shape[0])
    X_train_vectors = X_train_vectors[:sub_training_set_size]
    y_train = y_train[:sub_training_set_size]

    if model == "Naive Bayes":
        clf = GaussianNB()
        clf = clf.fit(X_train_vectors, y_train)
        prediction_train = clf.predict(X_train_vectors)
        prediction_test = clf.predict(X_test_vectors)
    elif model == "Logistic Regression":
        clf = LogisticRegression(C=params['C'], random_state=0, solver='lbfgs')
        clf = clf.fit(X_train_vectors, y_train)
        prediction_train = clf.predict(X_train_vectors)
        prediction_test = clf.predict(X_test_vectors)
    elif model == "Random Forest":
        clf = RandomForestClassifier(n_estimators=params['n_estimators'], max_depth=params['max_depth'],
                                     min_samples_leaf=params['min_samples_leaf'],
                                     min_samples_split=params['min_samples_split'],
                                     random_state=0, n_jobs=-1)
        clf = clf.fit(X_train_vectors, y_train)
        prediction_train = clf.predict(X_train_vectors)
        prediction_test = clf.predict(X_test_vectors)
    elif model == "SVM":
        clf = svm.SVC(C=params['C'], kernel=params['kernel'], random_state=0)
        clf = clf.fit(X_train_vectors, y_train)
        prediction_train = clf.predict(X_train_vectors)
        prediction_test = clf.predict(X_test_vectors)
    elif model == "XGBoost":
        clf = xgb.XGBClassifier(n_estimators=params['n_estimators'], max_depth=params['max_depth'],
                                min_child_weight=params['min_child_weight'], gamma=params['gamma'],
                                subsample=params['subsample'], colsample_bytree=params['colsample_bytree'],
                                reg_alpha=params['reg_alpha'], reg_lambda=params['reg_lambda'],
                                learning_rate=params['learning_rate'], random_state=0, n_jobs=-1).fit(X_train_vectors,
                                                                                                      y_train)  # https://xgboost.readthedocs.io/en/latest/tutorials/param_tuning.html
        prediction_train = clf.predict(X_train_vectors)
        prediction_test = clf.predict(X_test_vectors)
    elif model == "KNN":
        clf = KNeighborsClassifier(n_neighbors=params['n'])
        clf = clf.fit(X_train_vectors, y_train)
        prediction_train = clf.predict(X_train_vectors)
        prediction_test = clf.predict(X_test_vectors)
    else:
        clf = MLPClassifier(hidden_layer_sizes=params['hidden_layer_sizes'], random_state=0,
                            max_iter=params['max_iter'], learning_rate_init=params['learning_rate_init']).fit(
            X_train_vectors, y_train)
        prediction_train = clf.predict(X_train_vectors)
        prediction_test = clf.predict(X_test_vectors)

    accuracy_train = accuracy_score(prediction_train, y_train)
    accuracy_test = accuracy_score(prediction_test, y_test)
    f1_micro_train = f1_score(prediction_train, y_train, average='micro')
    f1_micro_test = f1_score(prediction_test, y_test, average='micro')
    f1_macro_train = f1_score(prediction_train, y_train, average='macro')
    f1_macro_test = f1_score(prediction_test, y_test, average='macro')

    train_score = {'accuracy': accuracy_train, 'f1_micro': f1_micro_train, 'f1_macro': f1_macro_train}
    print('training set performance: accuracy ', "{:.3f}".format(accuracy_train),
          'f1_micro ', "{:.3f}".format(f1_micro_train),
          'f1_macro ', "{:.3f}".format(f1_macro_train))

    test_score = {'accuracy': accuracy_test, 'f1_micro': f1_micro_test, 'f1_macro': f1_macro_test}
    print('test set performance: accuracy ', "{:.3f}".format(accuracy_test),
          'f1_micro ', "{:.3f}".format(f1_micro_test),
          'f1_macro ', "{:.3f}".format(f1_macro_test))

    return clf, prediction_train, prediction_test, train_score, test_score, X_train_vectors, X_test_vectors

stemmer = PorterStemmer()
# set working directory
path = '/Users/yhu245/Dropbox/CS584-textmining/Assignment/Assignment 2'
os.chdir(path)

# Load the data
f_path = './pdfalls.csv'
data = load_data_as_data_frame(f_path)
y = data['fall_class']
y = np.array([1 if x == 'CoM' else 0 for x in y])
X = data.drop('fall_class', axis=1)

# feature engineering MetaMap
meta_score = []
meta_concept = []
meta_semtypes = []
for i in range(X['fall_description'].shape[0]):
    print(i)
    concepts, error = mm.extract_concepts([X['fall_description'][i]])
    total_score = 0
    total_concept = 0
    total_semtypes = []
    for c in concepts:
        if c[1] == 'MMI':
            total_concept = total_concept + 1
            total_score = total_score + float(c.score)
            total_semtypes.append((str(c.semtypes)[1:-1]))

    meta_score.append(total_score)
    meta_concept.append(total_concept)
    meta_semtypes.append(total_semtypes)
meta_semtypes_count = []
for item in meta_semtypes:
    meta_semtypes_count.append(len(set(item)))

X_MetaMap = pd.DataFrame(np.array([meta_semtypes_count, meta_score, meta_concept]).transpose(), columns= ["semtype_count", 'score', 'concept'])
X = pd.concat([X, X_MetaMap], axis=1)

# train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
# deal with non-text feature
X_train_other = X_train[['record_id', 'age', 'female', 'duration', 'fall_study_day', 'semtype_count', 'avg_score']]
X_train_other['female'].replace(['Female','Male'], [0,1], inplace=True)
X_train_other = X_train_other.reset_index()
X_train_other.head()
X_train_other = X_train_other.drop('index', axis=1)
X_test_other = X_test[['record_id', 'age', 'female', 'duration', 'fall_study_day', 'semtype_count', 'avg_score']]
X_test_other['female'].replace(['Female','Male'], [0,1], inplace=True)
X_test_other = X_test_other.reset_index()
X_test_other.head()
X_test_other = X_test_other.drop('index', axis=1)
X_train_other.columns = ['Record_id', 'Age', 'Female', 'Duration', 'Fall_study_day', 'Semtype_count', 'Avg_score']
X_test_other.columns = ['Record_id', 'Age', 'Female', 'Duration', 'Fall_study_day', 'Semtype_count', 'Avg_score']

# transform text data into vectors
X_train_text_description = X_train['fall_description']
X_train_text_fall_location = X_train['fall_location']
X_test_text_description = X_test['fall_description']
X_test_text_fall_location = X_test['fall_location']
X_train_text_description_preprocessed = [preprocess_text(tr) for tr in X_train_text_description]
X_test_text_description_preprocessed = [preprocess_text(tr) for tr in X_test_text_description]
vectorizer = CountVectorizer(ngram_range=(1, 3), analyzer='word', tokenizer=None, preprocessor=None, max_features=10000, stop_words='english')
X_train_text_data_vectors = vectorizer.fit_transform(X_train_text_description_preprocessed).toarray()
X_test_text_data_vectors = vectorizer.transform(X_test_text_description_preprocessed).toarray()
X_train_text_data_vectors = pd.DataFrame(X_train_text_data_vectors, columns=vectorizer.get_feature_names())
X_test_text_data_vectors = pd.DataFrame(X_test_text_data_vectors, columns=vectorizer.get_feature_names())

# feature engineering text data
X_train_sent_length = []
X_test_sent_length = []
for tr in X_train_text_description:
    sent = sent_tokenize(tr)
    X_train_sent_length.append(len(sent))
for tr in X_test_text_description:
    sent = sent_tokenize(tr)
    X_test_sent_length.append(len(sent))

X_train_word_count = []
X_test_word_count = []
X_train_lexical_diversity = []
X_test_lexical_diversity = []

for tr in X_train_text_description_preprocessed:
    word = word_tokenize(tr)
    X_train_word_count.append(len(word))
    X_train_lexical_diversity.append(len(set(word)) / len(word))

for tr in X_test_text_description_preprocessed:
    word = word_tokenize(tr)
    X_test_word_count.append(len(word))
    X_test_lexical_diversity.append(len(set(word)) / len(word))

X_train_engineered = np.transpose(np.concatenate(([X_train_sent_length], [X_train_word_count], [X_train_lexical_diversity]), axis = 0))
X_train_engineered = pd.DataFrame(X_train_engineered, columns=['sent_length', 'word_count', 'lexical_diversity'])
X_test_engineered = np.transpose(np.concatenate(([X_test_sent_length], [X_test_word_count], [X_test_lexical_diversity]), axis = 0))
X_test_engineered = pd.DataFrame(X_test_engineered, columns=['sent_length', 'word_count', 'lexical_diversity'])
X_train_engineered_all = pd.concat([X_train_engineered, X_train_other], axis=1)
X_test_engineered_all = pd.concat([X_test_engineered, X_test_other], axis=1)

# selected engineered feature
include_columns = ['sent_length', 'word_count', 'lexical_diversity', 'Semtype_count', 'Avg_score']

# Feature Selection
# XGBoost Model
params_xgboost = {"n_estimators":100, "max_depth":6, "min_child_weight":1, "gamma":0,
                  "subsample":0.8, "colsample_bytree":0.8,
                  "reg_alpha":0, "reg_lambda":0, "learning_rate":0.3}
XGBoost_fit = ModelFit_Engineer_Feature(X_train_text_description_preprocessed, y_train, X_test_text_description_preprocessed,
                                        y_test, params=params_xgboost, model="XGBoost", vectorizer=vectorizer,
                                        X_train_engineered=X_train_engineered_all[include_columns],
                                        X_test_engineered=X_test_engineered_all[include_columns], feature_set=None)

def feature_selection(model, thresh):
    # select features using threshold
    selection = SelectFromModel(model[0], threshold=thresh, prefit= False)
    selection = selection.fit(model[5], y_train)
    selected_feat= model[5].columns[(selection.get_support())]
    select_X_train = model[5][selected_feat]
    # train model
    selection_model = selection.estimator
    selection_model.fit(select_X_train, y_train)
    # eval model
    select_X_test = model[6][selected_feat]
    #predictions = selection_model.predict(select_X_test)
    accuracy = accuracy_score(y_train, selection_model.predict(select_X_train))
    f1_score_micro = f1_score(y_train, selection_model.predict(select_X_train), average = 'micro')
    f1_score_macro = f1_score(y_train, selection_model.predict(select_X_train), average = 'macro')
    print("Thresh=%.3f, n=%d, Accuracy: %.3f%%" % (thresh, select_X_train.shape[1], accuracy))
    print("Thresh=%.3f, n=%d, f1_micro: %.3f%%" % (thresh, select_X_train.shape[1], f1_score_micro))
    print("Thresh=%.3f, n=%d, f1_macro: %.3f%%" % (thresh, select_X_train.shape[1], f1_score_macro))
    return selected_feat, accuracy, f1_score_micro, f1_score_macro

f1_score_micro_seq = {"threshold": [], "error": []}
for thresh in [0.00005, 0.001, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05]:
    print(thresh)
    selected_feature, accuracy_sub, f1_score_micro, f1_score_macro = feature_selection(XGBoost_fit, thresh = thresh)
    f1_score_micro_seq["threshold"].append(thresh)
    f1_score_micro_seq["error"].append(f1_score_micro)

f1_score_micro_seq = pd.DataFrame(f1_score_micro_seq, columns=['threshold', 'error'])
ax = plt.gca()
f1_score_micro_seq.plot(kind='line', x='threshold', y='error', ax=ax)
plt.ylabel("Training Micro-averaged F1 Score")
plt.xlabel("Threshold for Features")
plt.title("Training Micro-averaged F1 Score v.s. Different Subset of Features")
plt.show()
xgb.plot_importance(XGBoost_fit[0], importance_type='gain', max_num_features=24)
selected_feature, accuracy_sub, f1_score_micro, f1_score_macro = feature_selection(XGBoost_fit, thresh=0.01)


# cross validation for different classifier
skf = StratifiedKFold(n_splits=5)

## GNB
params ={'C':0}
GNBGridSearch(params, skf, train_data = X_train_text_description_preprocessed, train_label = y_train, train_engineer = X_train_engineered_all[include_columns], feature_set = selected_feature)

## KNN
params_knn = {'n':[3, 5, 7, 9, 10, 11, 13, 15]}
best_param_knn, best_metric_knn = KNNGridSearch(params_knn, skf, train_data = X_train_text_description_preprocessed, train_label = y_train,train_engineer = X_train_engineered_all[include_columns], feature_set = selected_feature)

## SVM
params_svm = {'C':[1, 2, 4, 8, 16, 32, 64], 'kernel':['linear', 'rbf']}
best_param_svm, best_metric_svm = SVMGridSearch(params, skf, train_data = X_train_text_description_preprocessed, train_label = y_train, train_engineer = X_train_engineered_all[include_columns], feature_set = selected_feature)

## Logistic Regression
params_lr = {'C':[1, 5, 10, 50, 100, 1000, 10000]}
best_param_lr, best_metric_lr = LogisticRegressionGridSearch(params_lr, skf, train_data=X_train_text_description_preprocessed, train_label=y_train,train_engineer = X_train_engineered_all[include_columns], feature_set = selected_feature)

## Random Forest
params_rf = {'n_estimators': [100, 200],
          'max_depth': [None, 4, 8],
          'min_samples_split': [2, 4],
          'min_samples_leaf': [1, 2]
          }
best_param_rf, best_metric_rf = RandomForestGridSearch(params_rf, skf, train_data = X_train_text_description_preprocessed, train_label = y_train, train_engineer = X_train_engineered_all[include_columns], feature_set = selected_feature)

## XGBoost
params_xgboost = {"n_estimators":[100, 200], "max_depth":[4, 6], "min_child_weight":[1], "gamma":[0],
                  "subsample":[0.6, 0.8], "colsample_bytree":[0.6, 0.8],
                  "reg_alpha":[0], "reg_lambda":[0], "learning_rate":[0.001, 0.005, 0.01, 0.05, 0.1]}
best_param_xgboost, best_metric_xgboost = XGBGridSearch(params = params_xgboost, skf = skf, train_data = X_train_text_description_preprocessed, train_label = y_train, train_engineer = X_train_engineered_all[include_columns], feature_set = selected_feature)

# Model Fitting
## GNB
params_gnb = {'C': 0}
gnb_fit = ModelFit_Engineer_Feature(X_train_text_description_preprocessed, y_train,
         X_test_text_description_preprocessed, y_test, params = params_knn, model = "Naive Bayes", vectorizer=vectorizer, X_train_engineered = X_train_engineered_all[include_columns], X_test_engineered = X_test_engineered_all[include_columns], feature_set=selected_feature)

## KNN
params_knn = {'n': 10}
knn_fit = ModelFit_Engineer_Feature(X_train_text_description_preprocessed, y_train,
         X_test_text_description_preprocessed, y_test, params=params_knn, model = "KNN", vectorizer=vectorizer, X_train_engineered = X_train_engineered_all[include_columns], X_test_engineered = X_test_engineered_all[include_columns], feature_set=selected_feature)

## SVM
params_svm = {'C': 1, 'kernel': 'linear'}
svm_fit = ModelFit_Engineer_Feature(X_train_text_description_preprocessed, y_train,
         X_test_text_description_preprocessed, y_test, params=params_svm, model="SVM", vectorizer=vectorizer, X_train_engineered = X_train_engineered_all[include_columns], X_test_engineered=X_test_engineered_all[include_columns], feature_set=selected_feature)

## Random Forest
params_rf = {'n_estimators': 100, 'max_depth': None, 'min_samples_split': 2,'min_samples_leaf': 1}
rf_fit = ModelFit_Engineer_Feature(X_train_text_description_preprocessed, y_train,
         X_test_text_description_preprocessed, y_test, params=params_rf, model="Random Forest", vectorizer=vectorizer, X_train_engineered = X_train_engineered_all[include_columns], X_test_engineered=X_test_engineered_all[include_columns], feature_set=selected_feature)

## Logistic Regression
params_lr = {'C': 10}
lr_fit = ModelFit_Engineer_Feature(X_train_text_description_preprocessed, y_train,
         X_test_text_description_preprocessed, y_test, params = params_lr, model="Logistic Regression", vectorizer=vectorizer,
         X_train_engineered=X_train_engineered_all[include_columns], X_test_engineered=X_test_engineered_all[include_columns], feature_set=selected_feature)

## XGBoost
params_xgb = best_param_XGBoost
xgb_fit = ModelFit_Engineer_Feature(X_train_text_description_preprocessed, y_train,
         X_test_text_description_preprocessed, y_test, params=params_xgb, model="XGBoost", vectorizer=vectorizer,
         X_train_engineered=X_train_engineered_all[include_columns], X_test_engineered=X_test_engineered_all[include_columns], feature_set=selected_feature)

## Ensemble
ensemble_fit_train = [1 if x>=2 else 0 for x in np.concatenate(([rf_fit[1]], [svm_fit[1]], [lr_fit[1]]), axis = 0).transpose().sum(axis =1)]
accuracy = accuracy_score(ensemble_fit_train, y_train)
f1_micro = f1_score(ensemble_fit_train, y_train, average='micro')
f1_macro = f1_score(ensemble_fit_train, y_train, average='macro')
print('accuracy', accuracy)
print('f1_micro', f1_micro)
print('f1_macro', f1_macro)
ensemble_fit = [1 if x>=2 else 0 for x in np.concatenate(([rf_fit[2]], [svm_fit[2]], [lr_fit[2]]), axis = 0).transpose().sum(axis =1)]
accuracy = accuracy_score(ensemble_fit, y_test)
f1_micro = f1_score(ensemble_fit, y_test, average='micro')
f1_macro = f1_score(ensemble_fit, y_test, average='macro')
print('accuracy', accuracy)
print('f1_micro', f1_micro)
print('f1_macro', f1_macro)

# Learning curve for different sample size
params_lr = {'C':5}
performance = {'p':[], 'micro f1_score':[], 'macro f1_score':[], 'accuracy':[]}
for p in range(1, 11, 1):
    p = p/10
    lr_fit = ModelFit_Engineer_Feature(X_train_text_description_preprocessed, y_train,
         X_test_text_description_preprocessed, y_test, params=params_lr, model="Logistic Regression", vectorizer=vectorizer,
         X_train_engineered=X_train_engineered_all[include_columns], X_test_engineered=X_test_engineered_all[include_columns], feature_set=selected_feature, p=p )
    performance['p'].append(p)
    performance['micro f1_score'].append(lr_fit[4]['f1_micro'])
    performance['macro f1_score'].append(lr_fit[4]['f1_macro'])
    performance['accuracy'].append(lr_fit[4]['accuracy'])

performance = pd.DataFrame(performance)
ax = plt.gca()
performance.plot(kind='line',x='p',y='micro f1_score',ax=ax)
performance.plot(kind='line',x='p',y='macro f1_score', color='black', ax=ax)
plt.ylabel("Score Value")
plt.xlabel("Proportion of Training Sample Size")
plt.title("Learning Curve with Different Training Sample Size")
plt.show()

# abelation Study
feature_performance = {'feature':[], 'f1_micro':[], 'f1_macro': [], 'accuracy': []}
for f in selected_feature:
    feature_tmp = selected_feature.copy()
    feature_tmp = feature_tmp.drop(f)
    print(len(feature_tmp))
    print(f)
    lr_fit = ModelFit_Engineer_Feature(X_train_text_description_preprocessed, y_train,
              X_test_text_description_preprocessed, y_test, params=params_lr, model="Logistic Regression", vectorizer=vectorizer,
              X_train_engineered=X_train_engineered_all[include_columns], X_test_engineered=X_test_engineered_all[include_columns], feature_set=feature_tmp)
    feature_performance['feature'].append(f)
    feature_performance['f1_micro'].append(lr_fit[4]['f1_micro'])
    feature_performance['f1_macro'].append(lr_fit[4]['f1_macro'])
    feature_performance['accuracy'].append(lr_fit[4]['accuracy'])

feature_performance = pd.DataFrame(feature_performance)
ax = plt.gca()
feature_performance.plot(kind='bar',x='feature',y='f1_micro',ax=ax)
plt.ylabel("Micro-averaged F1 Score")
plt.xlabel("Removed Feature")
plt.title("Ablation Study Results")
plt.show()