class Tagger:
  def __init__(self):
    from models.stop_words import make_stop_words, make_remove_adjectives, make_stop_words_tags
    self.stop_words_tags = make_stop_words_tags()
    self.stop_words = make_stop_words()
    self.remove_adjectives = make_remove_adjectives()
    
    from nltk.stem import WordNetLemmatizer
    self.lemmatizer = WordNetLemmatizer()

    self.good_chars = set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 -_")

    import spacy
    self.nlp = spacy.load('en_core_web_sm')

  def preprocess_abstract(self, abstract):
    import re
    okay = set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789,;.!?:'\"/\\$()[]{}- ")
    abstract = abstract.replace('$K$-theory', 'k-theory').replace('$C^*$-algebra', 'C-algebra').replace('\\emph', '').replace('\\emph{', '').replace('\\texit{', '')
    abstract = ' '.join([word for word in abstract.split(' ') if set([c for c in word]).issubset(okay)])
    #abstract = ''.join([c for c in abstract if c in okay])
    abstract = abstract.replace('\n', ' ') #remove new line characters
    abstract = re.sub(r'\\\S+', '', abstract) #remove anything after a slash
    abstract = abstract.replace('ä', 'a').replace('ö', 'o').replace('é', 'e')
    
    abstract = re.sub('\$.*?\$', '', abstract)
    abstract = abstract.replace('such a', ' ').replace('previously known', ' ').replace('so called', ' ').replace('more general', ' ').replace('all the', ' ').replace('all these', ' ').replace('very challenging', ' ')
    abstract = abstract.replace('so-called', ' ').replace('well known', ' ').replace('particularly nice', ' ')
    abstract = abstract.replace('"', '').replace("'", '').replace('`','').replace('\\', '').replace('--', '-').replace('^*', '')
    abstract = re.sub('\[.*?\]', '', abstract)
    abstract = re.sub('\s[a-zA-Z]{1}[\.\,\;]?\s', '. ', abstract)
    abstract = re.sub('\s[0-9]+\s', ' ', abstract)
    abstract = re.sub('\(.*?\)', '', abstract)
    abstract = abstract.replace('*', '').replace('{', '').replace('}', '')
    abstract = re.sub(' +', ' ', abstract)
    return abstract

  def shorten_abstract(self,abstract):
    doc = self.nlp(abstract)
    shortened = []
    for chunk in doc.noun_chunks:
        if ((doc[chunk.start].text in self.remove_adjectives) or (doc[chunk.start].pos_ in ['PRON', 'DET', 'INTJ', 'AUX', 'CCONJ', 'APD', 'NUM', 'PART', 'SCONJ', 'PUNCT', 'SYM', 'X'])) and (doc[chunk.end - 1].pos_ in ['PRON', 'DET', 'INTJ', 'AUX', 'CCONJ', 'APD', 'NUM', 'PART', 'SCONJ', 'PUNCT', 'SYM', 'X']):
            shortened.append('_'.join(chunk.text.split(' ')[1:-1]))

        elif ((doc[chunk.start].text in self.remove_adjectives) or (doc[chunk.start].pos_ in ['PRON', 'DET', 'INTJ', 'AUX', 'CCONJ', 'APD', 'NUM', 'PART', 'SCONJ', 'PUNCT', 'SYM', 'X'])):
            shortened.append('_'.join(chunk.text.split(' ')[1:]))

        elif (doc[chunk.end - 1].pos_ in ['PRON', 'DET', 'INTJ', 'AUX', 'CCONJ', 'APD', 'NUM', 'PART', 'SCONJ', 'PUNCT', 'SYM', 'X']):
            shortened.append('_'.join(chunk.text.split(' ')[:-1]))

        else: 
            shortened.append('_'.join(chunk.text.split(' ')))
    return ' '.join(shortened).strip()

  def get_ngrams(self,tokens, n=2):
    ngrams = zip(*[tokens[i:] for i in range(n)])
    ngrams = [' '.join(ngram) for ngram in ngrams]
    return ngrams

  def tokenize(self, text, ngram_range=(1,1)):
    import re
    tokens = re.findall(r'[a-z0-9_\'-]+', text.lower())
    ngrams = []
    for n in range(ngram_range[0], ngram_range[1]+1):
        ngrams += self.get_ngrams(tokens, n)
    return ngrams

  def remove_stopwords_and_lemmatize(self, text):
    from nltk.tokenize import word_tokenize 
    return ' '.join([self.lemmatizer.lemmatize(w) for w in word_tokenize(text) if w.lower() not in self.stop_words])

  def label_tokenizer(self,abstract):
    shortened = self.shorten_abstract(self.remove_stopwords_and_lemmatize(self.preprocess_abstract(abstract)))
    cleaned = ''.join([c for c in shortened if c in self.good_chars])
    return self.tokenize(cleaned)
    
  def generate_tags(self, abstract):
    tags = set([x.replace('_', ' ').strip() for x in self.label_tokenizer(abstract) if x not in self.stop_words_tags])
    return tags
