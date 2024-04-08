import math
import os
import nltk
from nltk.corpus import stopwords
import string
import numpy
from natsort import natsorted


# reads the contents of a text file
def readfile(txt_file):
    with open(txt_file, "r", encoding="ascii", errors="surrogateescape") as f:
        return f.read()


# tokenizes text, turns it into lower case, removes stop words and unattached punctuations.
def preprocessing(text_pre):
    token_list = nltk.word_tokenize(text_pre)
    token_list = [word.lower() for word in token_list]
    token_list = list(filter(lambda token: token not in string.punctuation, token_list))
    our_stopwords = stopwords.words('english')
    our_stopwords.remove('in')
    our_stopwords.remove('to')
    our_stopwords.remove('where')
    token_list = list(filter(lambda token: token not in our_stopwords, token_list))
    return token_list


# reads all the documents in our collection to build the positional index
# documents denoted as numbers in the positional index, but their names are kept in file_map for later use
def build_positional_index():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    file_names = natsorted(os.listdir(dir_path + r"\DocumentCollection"))
    positional_index = {}
    file_map = {}
    file_no = 1

    for filename in file_names:

        text = readfile(dir_path + r"\DocumentCollection\\" + filename)
        tokens = preprocessing(text)

        for pos, term in enumerate(tokens):

            if term in positional_index:
                if file_no in positional_index[term][1]:
                    positional_index[term][1][file_no].append(pos)
                else:
                    positional_index[term][1][file_no] = [pos]
                    positional_index[term][0] += 1
            else:
                positional_index[term] = [1]
                positional_index[term].append({file_no: [pos]})

        file_map[file_no] = filename
        file_no += 1

    return positional_index, file_map


# calculations necessary for the tf.idf matrix
def calc_idf(df, N):

    idf = math.log10(N / df)
    return idf


def calc_tf(raw_tf):

    tf = 1 + math.log10(raw_tf)
    return tf


def calc_tf_list(doc_dict, N):

    tf_list = [0] * N
    for i in range(N):
        if i+1 in doc_dict:
            raw_tf = len(doc_dict[i+1])
            tf = calc_tf(raw_tf)
            tf_list[i] = tf

    return tf_list


def calc_tfidf(tf_list, idf):

    tfidf = [round(element * idf, 5) for element in tf_list]
    return tfidf


# builds the tf.idf matrix using previous functions
def build_tfidf_matrix(positional_index, N):

    tfidf_matrix = {}
    for term in positional_index:
        idf = calc_idf(positional_index[term][0], N)
        tf_list = calc_tf_list(positional_index[term][1], N)
        tfidf = calc_tfidf(tf_list, idf)
        tfidf_matrix[term] = tfidf

    return tfidf_matrix


# calculating documents lengths to be used for normalization
def calc_doc_length(tfidf_matrix, N):

    doc_length = [0.0] * N
    for term in tfidf_matrix:
        doc_length = numpy.add(doc_length, [math.pow(element, 2) for element in tfidf_matrix[term]])
    doc_length = [math.sqrt(element) for element in doc_length]

    return doc_length


# traversing the positional index to get matched documents
def search_index(query, positional_index):

    matched_docs_sets = []

    if len(query) == 1:

        if query[0] in positional_index:
            result = [key for key in positional_index[query[0]][1]]
        else:
            result = []

        return result

    else:

        for i in range(len(query)-1):
            matched_docs = []
            if query[i] in positional_index:
                first_docs = positional_index[query[i]][1]
                if query[i+1] in positional_index:
                    second_docs = positional_index[query[i+1]][1]
                    for doc in first_docs:
                        if doc in second_docs:
                            for pos in first_docs[doc]:
                                if pos + 1 in second_docs[doc]:
                                    matched_docs.append(doc)
                else:
                    break
            else:
                break

            if matched_docs:
                matched_docs_sets.append(set(matched_docs))
            else:
                result = []
                return result
    if matched_docs_sets:
        result = set.intersection(*matched_docs_sets)
    else:
        return []
    return list(result)


# query calculations
def query_rtf(token_list):

    rtf = {}
    for token in token_list:
        if token in rtf:
            rtf[token] += 1
        else:
            rtf[token] = 1
    return rtf


def query_tf(rtf):

    tf = {}
    for token in rtf:
        tf[token] = calc_tf(rtf[token])
    return tf


def query_tfidf(qtf, pos_index, N):

    tfidf = {}
    for term in qtf:
        if term in pos_index:
            idf = calc_idf(pos_index[term][0], N)
        else:
            idf = 0
        tfidf[term] = round(idf * qtf[term], 5)
    return tfidf


def query_length(q_tfidf):

    length = 0.0
    for term in q_tfidf:
        length += math.pow(q_tfidf[term], 2)
    length = math.sqrt(length)
    return length


def query_normalize(length, q_tfidf):

    q_normalized = {}
    for term in q_tfidf:
        if length == 0:
            q_normalized[term] = 0
        else:
            q_normalized[term] = q_tfidf[term] / length
    return q_normalized


# calculates normalized scores for query terms in matched documents from our collection
def doc_normalize(doc_no, doc_lengths, tfidf_matrix, token_list):

    doc_normalized = {}
    for token in token_list:
        if token in tfidf_matrix:
            doc_normalized[token] = tfidf_matrix[token][doc_no-1] / doc_lengths[doc_no-1]
        else:
            doc_normalized[token] = 0
    return doc_normalized


# applying our calculations to the query, positional index needed for idf
def query_processing(query, positional_index, N):

    token_list = preprocessing(query)
    rtf = query_rtf(token_list)
    tf = query_tf(rtf)
    tfidf = query_tfidf(tf, positional_index, N)
    length = query_length(tfidf)
    query_normalized = query_normalize(length, tfidf)
    return token_list, query_normalized


# calculates document score using normalized weights for terms in query and document collection
def doc_score(q_normalized, d_normalized):

    score = 0.0
    for term in q_normalized:
        score += q_normalized[term] * d_normalized[term]
    return score


# prints the positional index and the tf.idf matrix
def show_data(positionalindex, tfidf_matrix, file_map, normalized_terms):

    print("----------------------------------------")
    print("Positional Index")
    for term in positionalindex:
        print(term, positionalindex[term])
    print("----------------------------------------")
    print("TF.IDF Matrix")
    print("   ", end="")
    for x in file_map:
        print(" ", file_map[x], end="")
    print("")
    for term in tfidf_matrix:
        print(term, tfidf_matrix[term])
    print("----------------------------------------")
    print("Normalized tf.id")
    for term in normalized_terms:
        print(term, normalized_terms[term])
    print("----------------------------------------")


# utilizing everything we built to return the matched documents, ranked
def engine(query, positionalindex, file_map, tfidf_matrix, doc_lengths, N):
    docs = {}
    token_list = query_processing(query, positionalindex, N)[0]
    query_normalized = query_processing(query, positionalindex, N)[1]
    matched_docs = search_index(token_list, positionalindex)
    for i in range(len(matched_docs)):
        doc_normalized = doc_normalize(matched_docs[i], doc_lengths, tfidf_matrix, token_list)
        docs[matched_docs[i]] = doc_score(query_normalized, doc_normalized)
    ranked_results = {}
    sorted_keys = sorted(docs, key=docs.get, reverse=True)
    for key in sorted_keys:
        ranked_results[file_map[key]] = docs[key]
    if ranked_results == {}:
        return {}
    else:
        return ranked_results


def normalize_terms(tfidf_matrix, doc_length):
    normalized_matrix = {}
    for term in tfidf_matrix:
        normalized_matrix[term] = []
        i = 0
        for element in tfidf_matrix[term]:
            normalized_matrix[term].append(round((element / doc_length[i]), 5))
            i = i + 1
    return normalized_matrix
