import re 
import gensim
from gensim.utils import simple_preprocess
from word_forms.lemmatizer import lemmatize

def preprocess(data):
    #removing punctuation 
    data = data.map(lambda x: re.sub('[,/.!?]', '', x))

    # Convert to lowercase
    data = data.map(lambda x: x.lower())


    #removal of stop words 
    nltk_stopwords = open("NLTKStopwords.txt").readlines()
    nltk_stopwords = [x.strip('\n') for x in nltk_stopwords]

    stop_words = nltk_stopwords.copy()
    stop_words.extend(['admiral', 'insurance', 'take', 'tell', 'make', 'say', 'go', 'get', 'everything', 'na', "", 'bell', 'policy','car','diamond']) #specific stopwords 
    data = data.values.tolist()

    def sent_to_words(sentences):
        for sentence in sentences:
            yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations, converts documents into a list of tokens

    data_words = list(sent_to_words(data))

    def remove_stopwords(texts, stop_words):
        return [[word for word in doc if word not in stop_words] for doc in texts]

    data_words_nostops = remove_stopwords(data_words, stop_words)


    def lemmatization(texts):
        texts_out = []
        for i in texts:
            sent_lst = []
            for j in i:
                out = j
                try:
                    out = lemmatize(out)
                except:
                    pass
                sent_lst.append(out)
            texts_out.append(sent_lst)
        return texts_out

    data_lemmatized = lemmatization(data_words_nostops)

    #flattening the nested list created from previous steps 
    cleanlist = []

    for words in data_lemmatized:
        string = ' '.join(words)
        cleanlist.append(string)
        
    cleanlist = [re.sub('drift', 'drive', x) for x in cleanlist] #undoing lemmatization 
    cleanlist = [re.sub('err', 'error', x) for x in cleanlist]
    cleanlist = [re.sub('intuit', 'intuitive', x) for x in cleanlist]
        
    return cleanlist 