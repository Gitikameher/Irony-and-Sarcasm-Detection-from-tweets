#!/bin/python
import sys
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def read_files(tarfname):
    """Read the training and development data from the sentiment tar file.
    The returned object contains various fields that store sentiment data, such as:

    train_data,dev_data: array of documents (array of words)
    train_fnames,dev_fnames: list of filenames of the doccuments (same length as data)
    train_labels,dev_labels: the true string label for each document (same length as data)

    The data is also preprocessed for use with scikit-learn, as:

    count_vec: CountVectorizer used to process the data (for reapplication on new data)
    trainX,devX: array of vectors representing Bags of Words, i.e. documents processed through the vectorizer
    le: LabelEncoder, i.e. a mapper from string labels to ints (stored for reapplication)
    target_labels: List of labels (same order as used in le)
    trainy,devy: array of int labels, one for each document
    """
    import tarfile
    tar = tarfile.open(tarfname, "r:gz")
    trainname = "train.tsv"
    devname = "dev.tsv"
    for member in tar.getmembers():
        if 'train.tsv' in member.name:
            trainname = member.name
        elif 'dev.tsv' in member.name:
            devname = member.name
            
            
    class Data: pass
    sentiment = Data()
    print("-- train data")
    sentiment.train_data, sentiment.train_labels = read_tsv(tar,trainname)
    print(len(sentiment.train_data))

    print("-- dev data")
    sentiment.dev_data, sentiment.dev_labels = read_tsv(tar, devname)
    print(len(sentiment.dev_data))
    print("-- transforming data and labels")
    from sklearn.feature_extraction.text import CountVectorizer
    sentiment.count_vect = CountVectorizer()
    sentiment.trainX = sentiment.count_vect.fit_transform(sentiment.train_data)
    sentiment.devX = sentiment.count_vect.transform(sentiment.dev_data)
    from sklearn import preprocessing
    sentiment.le = preprocessing.LabelEncoder()
    sentiment.le.fit(sentiment.train_labels)
    sentiment.target_labels = sentiment.le.classes_
    sentiment.trainy = sentiment.le.transform(sentiment.train_labels)
    sentiment.devy = sentiment.le.transform(sentiment.dev_labels)
    tar.close()
    return sentiment

def read_unlabeled(tarfname, sentiment):
    """Reads the unlabeled data.

    The returned object contains three fields that represent the unlabeled data.

    data: documents, represented as sequence of words
    fnames: list of filenames, one for each document
    X: bag of word vector for each document, using the sentiment.vectorizer
    """
    import tarfile
    tar = tarfile.open(tarfname, "r:gz")
    class Data: pass
    unlabeled = Data()
    unlabeled.data = []
    
    unlabeledname = "unlabeled.tsv"
    for member in tar.getmembers():
        if 'unlabeled.tsv' in member.name:
            unlabeledname = member.name
            
    print(unlabeledname)
    tf = tar.extractfile(unlabeledname)
    for line in tf:
        line = line.decode("utf-8")
        text = line.strip()
        unlabeled.data.append(text)
        
            
    unlabeled.X = sentiment.count_vect.transform(unlabeled.data)
    print(unlabeled.X.shape)
    tar.close()
    return unlabeled

def read_tsv(tar, fname):
    member = tar.getmember(fname)
    print(member.name)
    tf = tar.extractfile(member)
    data = []
    labels = []
    for line in tf:
        line = line.decode("utf-8")
        (label,text) = line.strip().split("\t")
        labels.append(label)
        data.append(text)
    return data, labels

def write_pred_kaggle_file(unlabeled, cls, outfname, sentiment):
    """Writes the predictions in Kaggle format.

    Given the unlabeled object, classifier, outputfilename, and the sentiment object,
    this function write sthe predictions of the classifier on the unlabeled data and
    writes it to the outputfilename. The sentiment object is required to ensure
    consistent label names.
    """
    yp = cls.predict(unlabeled.X)
    labels = sentiment.le.inverse_transform(yp)
    f = open(outfname, 'w')
    f.write("ID,LABEL\n")
    for i in range(len(unlabeled.data)):
        f.write(str(i+1))
        f.write(",")
        f.write(labels[i])
        f.write("\n")
    f.close()


def write_gold_kaggle_file(tsvfile, outfname):
    """Writes the output Kaggle file of the truth.

    You will not be able to run this code, since the tsvfile is not
    accessible to you (it is the test labels).
    """
    f = open(outfname, 'w')
    f.write("ID,LABEL\n")
    i = 0
    with open(tsvfile, 'r') as tf:
        for line in tf:
            (label,review) = line.strip().split("\t")
            i += 1
            f.write(str(i))
            f.write(",")
            f.write(label)
            f.write("\n")
    f.close()

def write_basic_kaggle_file(tsvfile, outfname):
    """Writes the output Kaggle file of the naive baseline.

    This baseline predicts POSITIVE for all the instances.
    """
    f = open(outfname, 'w')
    f.write("ID,LABEL\n")
    i = 0
    with open(tsvfile, 'r') as tf:
        for line in tf:
            (label,review) = line.strip().split("\t")
            i += 1
            f.write(str(i))
            f.write(",")
            f.write("POSITIVE")
            f.write("\n")
    f.close()

def get_top(sentiment, cls):
    import numpy as np
    coefficients=cls.coef_[0]
    # k = 8
    # top_k =np.argsort(coefficients)[-k:]
    # top_k_words = []

    # print('-'*50)
    # print('Top k=%d' %k)
    # print('-'*50)

    # for i in top_k:
    #     print(sentiment.count_vect.get_feature_names()[i])
    #     top_k_words.append(sentiment.count_vect.get_feature_names()[i])
    # print(top_k_words)
    # print(top_k)
    # print(coefficients)
    # print('yoooo')
    # #print(sentiment.count_ve
    # print('-'*50)
    # print('Bottom k=%d' %k)
    # print('-'*50)
    # #top_k = np.argpartition(coefficients, -k)[-k:]
    # bottom_k =np.argsort(coefficients)[:k]
    # bottom_k_words = []
    # #print(top_k)
    # for i in bottom_k:
    #     print(sentiment.count_vect.get_feature_names()[i])
    #     bottom_k_words.append(sentiment.count_vect.get_feature_names()[i])
    print(len(coefficients))
    hashmap = dict()
    for i in range(len(coefficients)):
        word = sentiment.count_vect.get_feature_names()[i]
        hashmap[word] = coefficients[i]
    print (hashmap)
    return hashmap

def get_words(indices, sentiment):
    ret = []
    for i in indices:
        ret.append(sentiment.count_vect.get_feature_names()[i])
    return ret

def predict_sample(sample, cls, sentiment, hashmap):
    out = sentiment.count_vect.transform(sample)
    predictions = cls.predict_proba(out)
    indices = out[0].nonzero()[1]
    words = get_words(indices, sentiment)
    weights = [hashmap[w] for w in words]
    new_weights = [w * 100 for w in weights]
    print_pic(new_weights, 'Most positive predicting words', words)
    new_weights_negative = [w * -100 for w in weights]
    print_pic(new_weights_negative, 'Most negative predicting words', words)
    print(predictions)
    return (predictions, words, weights)

def print_pic(new_weights, title, words):
    sentence = []
    for i in range(len(new_weights)):
        w = new_weights[i]
        curr = int(w)
        while (curr > 0):
            sentence.append(words[i])
            curr-=1
    generate_wordcloud(' '.join(sentence), title)

def generate_wordcloud(text,title='Test Title'): # optionally add: stopwords=STOPWORDS and change the arg below
    fig=plt.figure()
    wordcloud = WordCloud(font_path='/Library/Fonts/Verdana.ttf',
                          relative_scaling = 1.0,
                          collocations=False,
                          stopwords = {'to', 'of'} # set or space-separated string
                          ).generate(text)
    plt.imshow(wordcloud)
    plt.title(title)
    plt.axis("off")
    #plt.show()
    pp.savefig(fig)

pp = PdfPages('foo2.pdf')

if __name__ == "__main__":
    sample = [str(sys.argv[1])]
    print(sample)
    print("Reading data")
    tarfname = "data/sentiment.tar.gz"
    sentiment = read_files(tarfname)
    print("\nTraining classifier")
    #sample = "Went last night for the first time with my boyfriend. Let me start off by saying I'm vegetarian, but my boyfriend is not. I ordered the chicken v mushroom, it"
    import classify
    cls = classify.train_classifier(sentiment.trainX, sentiment.trainy, 1)
    hashmap = get_top(sentiment, cls)
    print("\nEvaluating")
    classify.evaluate(sentiment.trainX, sentiment.trainy, cls, 'train')
    classify.evaluate(sentiment.devX, sentiment.devy, cls, 'dev')

    predictions, words, weights = predict_sample(sample, cls, sentiment, hashmap)
    
    from matplotlib.pyplot import figure
    fig2 = figure(num=None, figsize=(20, 16), dpi=80, facecolor='w', edgecolor='k')
    plt.title('Weights on words')
    plt.bar(words, weights)
    pp.savefig(fig2)

    fig = plt.figure()
    plt.pie(predictions[0], labels=['NEGATIVE', 'POSITIVE'], colors=['RED', 'GREEN'], autopct='%1.1f%%')
    plt.title('Confidence')
    pp.savefig(fig)

    print("\nReading unlabeled data")
    unlabeled = read_unlabeled(tarfname, sentiment)
    print("Writing predictions to a file")
    write_pred_kaggle_file(unlabeled, cls, "data/sentiment-pred.csv", sentiment)
    #write_basic_kaggle_file("data/sentiment-unlabeled.tsv", "data/sentiment-basic.csv")
    pp.close()
    import webbrowser
    webbrowser.open_new(r'foo2.pdf')
    # You can't run this since you do not have the true labels
    # print "Writing gold file"
    # write_gold_kaggle_file("data/sentiment-unlabeled.tsv", "data/sentiment-gold.csv")
