import os
import Levenshtein
import string
import re
import nltk
from nltk import word_tokenize, sent_tokenize
import itertools
import pandas as pd
from collections import defaultdict
import openpyxl

path = '/Users/yhu245/Dropbox/CS584-textmining/Assignment/Assignment 1/'
os.chdir(path)

def load_labels(f_path):
    '''
    Loads the labels

    :param f_path:
    :return:
    '''
    labeled_df = pd.read_excel(f_path)
    labeled_dict = defaultdict(list)
    for index, row in labeled_df.iterrows():
        id_ = row['ID']
        if not pd.isna(row['Symptom CUIs']) and not pd.isna(row['Negation Flag']):
            cuis = row['Symptom CUIs'].split('$$$')[1:-1]
            neg_flags = row['Negation Flag'].split('$$$')[1:-1]
            for cui, neg_flag in zip(cuis, neg_flags):
                labeled_dict[id_].append(cui + '-' + str(neg_flag))
    return labeled_dict

def run_sliding_window_through_text(words, window_size):
    """
    Generate a window sliding through a sequence of words
    """
    word_iterator = iter(words)  # creates an object which can be iterated one element at a time
    word_window = tuple(itertools.islice(word_iterator,
                                         window_size))  # islice() makes an iterator that returns selected elements from the the word_iterator
    yield word_window
    # now to move the window forward, one word at a time
    for w in word_iterator:
        word_window = word_window[1:] + (w,)
        yield word_window

def match_dict_similarity(text, expressions, threshold=0.8):
    '''
    :param text:
    :param expressions:
    :param threshold:
    :return:
    '''
    max_similarity_obtained = -1
    best_match = ''
    best_exp = ''
    # go through each expression
    for exp in expressions:
        # create the window size equal to the number of word in the expression in the lexicon
        size_of_window = len(exp.split())
        tokenized_text = list(nltk.word_tokenize(text))
        for window in run_sliding_window_through_text(tokenized_text, size_of_window):
            window_string = ' '.join(window)

            similarity_score = Levenshtein.ratio(window_string, exp)

            if similarity_score >= threshold:
                # print (similarity_score,'\t', exp,'\t', window_string)
                # print(1 - similarity_score/max(len(exp), len(window_string)), '\t', exp, '\t', window_string)
                if similarity_score > max_similarity_obtained:
                    max_similarity_obtained = similarity_score
                    best_match = window_string
                    best_exp = exp
    res = [best_exp, max_similarity_obtained, best_match]
    return res

def get_key(val, my_dict):
    res = []
    for key, value in my_dict.items():
        if val == value:
            res.append(key)
    return res

def auto_annotation(text, expressions, neg, threshold = 0.85):

    text_preprocessed = text
    for c in string.punctuation:
        text_preprocessed = text_preprocessed.replace(c, "")

    res = []
    word_string = {}
    for cui in set(expressions.values()):
        tmp = match_dict_similarity(text_preprocessed, get_key(cui, expressions), threshold = threshold)
        if tmp[1] > -1:
            if tmp[2] in word_string:
                if tmp[1] > word_string[tmp[2]]:
                    word_string[tmp[2]] = tmp[1]
                    tmp.append(cui)
                    res.append(tmp)
            else:
                word_string.update({tmp[2]: tmp[1]})
                tmp.append(cui)
                res.append(tmp)

    text2 = text
    for w in res:
        text2 = text2.replace(w[2], w[0])

    sentences = sent_tokenize(text2)
    output = []
    output_cui = '$$$'
    output_flag = '$$$'
    for sent in sentences:
        flag = 0
        for exp in expressions:
            tmp_exp = r'\b' + exp + r'\b'
            pat = re.compile(tmp_exp)
            match_object = re.search(pat, sent)
            if match_object:
                # print(match_object.group())
                for neg_w in neg:
                    pat = re.compile(r'\b' + neg_w + '\W+(?:\w+\W+){0,3}?' + exp + r'\b')
                    match_object_neg = re.search(pat, sent)
                    if match_object_neg:
                        flag = 1
                tmp = expressions[exp] + '-' + str(flag)
                output.append(tmp)
                output_cui = output_cui + expressions[exp] + '$$$'
                output_flag = output_flag + str(flag) + '$$$'
            tmp = {}
            flag = 0

    return output, res, output_flag, output_flag

def annotation(file, expressions, neg, threshold = 0.85):
    labeled_df = pd.read_excel(file)
    annotation = {}
    detect = {}
    cui = {}
    flag = {}
    for index, row in labeled_df.iterrows():
        id_ = row['ID']
        if not pd.isna(row['TEXT']):
            print(id_)
            text = row['TEXT']
            tmp_annotation, tmp_detect, tmp_cui, tmp_flag = auto_annotation(text, expressions, neg, threshold)
            annotation.update({id_:tmp_annotation})
            detect.update({id_:tmp_detect})
            cui.update({id_: tmp_cui})
            flag.update({id_: tmp_flag})
    return annotation, detect, cui, flag

def calculate_metric(submission_dict, gold_standard_dict):
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    tp_id = {}
    fp_id = {}
    fn_id = {}
    for k,v in gold_standard_dict.items():
        for c in v:
            try:
                if c in submission_dict[k]:
                    tp+=1
                    if k not in tp_id:
                        tp_id.update({k:[]})
                        tp_id[k].append(c)
                    else:
                        tp_id[k].append(c)
                else:
                    fn+=1
                    if k not in fn_id:
                        fn_id.update({k: []})
                        fn_id[k].append(c)
                    else:
                        fn_id[k].append(c)
            except KeyError:#if the key is not found in the submission file, each is considered
                            #to be a false negative..
                fn+=1
                if k not in fn_id:
                    fn_id.update({k: []})
                    fn_id[k].append(c)
                else:
                    fn_id[k].append(c)
        for c2 in submission_dict[k]:
            if not c2 in gold_standard_dict[k]:
                fp+=1
                if k not in fp_id:
                    fp_id.update({k: []})
                    fp_id[k].append(c2)
                else:
                    fp_id[k].append(c2)

    print('True Positives:',tp, 'False Positives: ', fp, 'False Negatives:', fn)
    recall = tp/(tp+fn)
    precision = tp/(tp+fp)
    f1 = (2*recall*precision)/(recall+precision)
    print('Recall: ',recall,'\nPrecision:',precision,'\nF1-Score:',f1)
    return f1, tp_id, fn_id, fp_id

def test_code(file, expressions, neg, file_gold_standard = '', output_file = '', train = True):
    submission_dict, annotation_list, annotation_cui, annotation_flag = annotation(file, expressions, neg)
    if train:
        gold_standard_dict = load_labels(file_gold_standard)
        f1_1, tp_id_1, fn_id_1, fp_id_1 = calculate_metric(submission_dict, gold_standard_dict)
        return f1_1, tp_id_1, fn_id_1, fp_id_1
    else:
        dict = {"ID": list(submission_dict.keys()), "Symptom CUIs": list(annotation_cui.values()),
                 "Negation Flag": list(annotation_flag.values())}
        df = pd.DataFrame.from_dict(data=dict)
        df = df.dropna()
        df.to_excel(output_file, index=False)
        return

if __name__ == "__main__":
    # neg lexicon
    infile = open('./neg_trigs.txt')
    neg = infile.readlines()
    neg = [x.strip('\n').strip('\t') for x in neg]

    # load lexicon
    infile = open('./COVID-Twitter-Symptom-Lexicon.txt')
    expressions = {}
    for line in infile:
        items = line.split('\t')
        expressions.update({str.strip(items[-1]): str.strip(items[1])})

    test_code('./Assignment1GoldStandardSet.xlsx', expressions, neg, './Assignment1GoldStandardSet.xlsx')
    #test_code('./UnlabeledSet2.xlsx', expressions, neg, output_file='Submission_UnlabeledSet2.xlsx', train = False)

# code for threshold chosen
#f1 = []
#for thres in [0.7, 0.75, 0.8, 0.85, 0.9]:
#    submission_dict, annotation_list, annotation_cui, annotation_flag= annotation('./Assignment1GoldStandardSet.xlsx', expressions = expressions, neg = neg, threshold=thres)
#    gold_standard_dict = load_labels('./Assignment1GoldStandardSet.xlsx')
#    f1_temp, tp_id_temp, fn_id_temp, fp_id_temp = calculate_metric(submission_dict, gold_standard_dict)
#    f1.append(f1_temp)




