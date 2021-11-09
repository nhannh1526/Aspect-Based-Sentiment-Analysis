import os
import pickle
import re

import pandas as pd
from gensim.utils import simple_preprocess
from nltk import flatten

from vncorenlp import VnCoreNLP

root_path = os.getcwd()
train_path = root_path + "/files/Train.txt"
dev_path = root_path + "/files/Dev.txt"
test_path = root_path + "/files/Test.txt"
pos_path = root_path + "/files/pos.txt"
neg_path = root_path + "/files/neg.txt"
replace_list_path = root_path + "/files/replace_list.pkl"
teen_code_path = root_path + "/files/teen_code.txt"
VnCoreNLP_path = root_path + "/vncorenlp/VnCoreNLP-1.1.1.jar"
data_path = root_path + "/data"

rdrsegmenter = VnCoreNLP(
    VnCoreNLP_path, annotators="wseg", max_heap_size="-Xmx500m")
REPLACE_LIST = pickle.load(open(replace_list_path, "rb"))
pickle.dump(REPLACE_LIST, open(replace_list_path, "wb"))
with open(pos_path, "r", encoding="utf-8") as file:
    POS_LIST = [i.strip() for i in file]
with open(neg_path, "r", encoding="utf-8") as file:
    NEG_LIST = [i.strip() for i in file]
teen_code = pd.read_csv(teen_code_path, sep="\t",
                        header=None, encoding="utf-8")
TEEN_CODE_LIST = dict(sorted(teen_code.values.tolist()))

VN_CHARS_LOWER = u'ạảãàáâậầấẩẫăắằặẳẵóòọõỏôộổỗồốơờớợởỡéèẻẹẽêếềệểễúùụủũưựữửừứíìịỉĩýỳỷỵỹđð'
VN_CHARS_UPPER = u'ẠẢÃÀÁÂẬẦẤẨẪĂẮẰẶẲẴÓÒỌÕỎÔỘỔỖỒỐƠỜỚỢỞỠÉÈẺẸẼÊẾỀỆỂỄÚÙỤỦŨƯỰỮỬỪỨÍÌỊỈĨÝỲỶỴỸÐĐ'
VN_CHARS = VN_CHARS_LOWER + VN_CHARS_UPPER


def get_no_diacritic(sentence):
    __INTAB = [ch for ch in VN_CHARS]
    __OUTTAB = "a" * 17 + "o" * 17 + "e" * 11 + \
               "u" * 11 + "i" * 5 + "y" * 5 + "d" * 2
    __OUTTAB += "A" * 17 + "O" * 17 + "E" * 11 + \
                "U" * 11 + "I" * 5 + "Y" * 5 + "D" * 2
    __r = re.compile("|".join(__INTAB))
    __replaces_dict = dict(zip(__INTAB, __OUTTAB))
    result = __r.sub(lambda m: __replaces_dict[m.group(0)], sentence)
    return result


class FeedBack:
    def __init__(self, file_path):
        self.file_path = file_path
        self.sentences = None
        self.no_diacritic_sentences = None
        self.opinions = None
        self.aspects = None
        self.polarities = None

    def get_sentences_opinions(self):
        print("[INFO] Get sentences and opinions...")
        with open(self.file_path, "r", encoding="utf-8") as f:
            data = f.readlines()
        get = 1
        sentences = []
        opinions = []
        for line in data:
            if get == 2:
                sentences.append(line)
                get += 1
            elif get == 3:
                opinions.append(line)
                get += 1
            elif get == 4:
                get = 1
            else:
                get += 1
        self.sentences = sentences
        self.opinions = opinions

    def get_aspects_polarities(self):
        print("[INFO] Get aspects and polarities...")
        aspects = []
        polarities = []
        for i in self.opinions:
            opinion = i.strip().replace("{", "").replace("}", "").split(", ")
            aspects.append(opinion[0:len(opinion):2])
            polarities.append([{category: polarity} for category, polarity in
                               zip(opinion[0:len(opinion):2], opinion[1:len(opinion):2])])
        self.aspects = aspects
        self.polarities = polarities

    def remove_elongate_chars(self):
        print("[INFO] Remove elongate characters...")
        sentences = []
        for sentence in self.sentences:
            check = re.search(r'([a-z])\1+', sentence)
            if check:
                if len(check.groups()) > 2:
                    sentence = re.sub(
                        r'([a-z])\1+', lambda m: m.group(1), sentence, flags=re.IGNORECASE)
            sentences.append(sentence)
        self.sentences = sentences

    def remove_leading_spaces(self):
        print("[INFO] Remove leading spaces...")
        sentences = []
        for sentence in self.sentences:
            sentence = sentence.strip()
            sentences.append(sentence)
        self.sentences = sentences

    def convert_lower_case(self):
        print("[INFO] Convert lower case...")
        sentences = []
        for sentence in self.sentences:
            sentence = sentence.lower()
            sentences.append(sentence)
        self.sentences = sentences

    def replace_char_in_list(self):
        print("[INFO] Replace characters in replace list...")
        sentences = []
        for sentence in self.sentences:
            for k, v in REPLACE_LIST.items():
                sentence = sentence.replace(k, v)
            # for k, v in TEEN_CODE_LIST.items():
            #     sentence = sentence.replace(k, v)
            sentences.append(sentence)
        self.sentences = sentences

    def remove_special_chars(self):
        print("[INFO] Remove special characters...")
        sentences = []
        for sentence in self.sentences:
            sentence = re.sub(r'< a class.+</a>', ' ', sentence)
            # special character
            sentence = re.sub(
                r'[!”"#$%&’()•/:;<=>-?@[\]^`{|}~+*_-]', ' ', sentence)
            sentence = re.sub(r'\d+k', '', sentence)  # 100k
            sentence = re.sub(r'\d+', '', sentence)  # number
            sentence = re.sub(
                r'(https|http)?:/(\w|\.|/|\?|=|&|%)*\b', ' ', sentence)  # web address
            sentence = re.sub(r'www\.\S+\.com', ' ', sentence)  # web address
            sentence = re.sub(r'@\S+', ' ', sentence)  # user mentioned
            sentence = re.sub(r'[0-9]k', ' ', sentence)
            emoji_pattern = re.compile("["
                                       u"\U0001F600-\U0001F64F"  # emoticons
                                       u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                       u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                       u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                                       u"\U00002702-\U000027B0"
                                       u"\U000024C2-\U0001F251"
                                       "]+", flags=re.UNICODE)
            sentence = re.sub(emoji_pattern, '', sentence)
            sentence = re.sub(r'_', ' ', sentence)
            sentences.append(sentence)
        self.sentences = sentences

    def tokenize(self):
        print("[INFO] Tokenizing...")
        sentences = []
        for sentence in self.sentences:
            sentence = ' '.join(i for i in flatten(
                rdrsegmenter.tokenize(sentence)))
            sentences.append(sentence)
        self.sentences = sentences

    def add_pos_neg_features(self):
        print("[INFO] Add POS and NEG features...")
        sentences = []
        for sentence in self.sentences:
            for i in POS_LIST:
                if re.search(' ' + i + ' ', sentence):
                    sentence = re.sub(i, i + ' positive ', sentence)
            for i in NEG_LIST:
                if re.search(' ' + i + ' ', sentence):
                    sentence = re.sub(i, i + ' negative ', sentence)
            sentences.append(sentence)
        self.sentences = sentences

    def get_no_diacritic_sentences(self):
        print("[INFO] Get no diacritic sentences...")
        sentences = []
        for sentence in self.sentences:
            sentence = get_no_diacritic(sentence=re.sub(r'_', ' ', sentence))
            sentences.append(sentence)
        self.no_diacritic_sentences = sentences

    def preprocess(self):
        print("[INFO] Preprocess by gensim library...")
        sentences = []
        for sentence in self.sentences:
            sentence = ' '.join(i for i in simple_preprocess(sentence))
            sentences.append(sentence)
        self.sentences = sentences

        no_diacritic_sentences = []
        for sentence in self.no_diacritic_sentences:
            sentence = ' '.join(i for i in simple_preprocess(sentence))
            no_diacritic_sentences.append(sentence)
        self.no_diacritic_sentences = no_diacritic_sentences

    def no_preprocess(self):
        sentences = []
        no_diacritic_sentences = []
        for sentence in self.sentences:
            sentence = ' '.join(i for i in flatten(
                rdrsegmenter.tokenize(sentence)))
            no_diacritic_sentence = get_no_diacritic(
                sentence=re.sub(r'_', ' ', sentence))
            sentences.append(sentence)
            no_diacritic_sentences.append(no_diacritic_sentence)
        self.sentences = sentences
        self.no_diacritic_sentences = no_diacritic_sentences


if __name__ == "__main__":
    if not os.path.exists(data_path):
        print("[INFO] Create data directory...")
        os.mkdir(data_path)

    print("[INFO] Prepare Training Set...")
    Train = FeedBack(train_path)
    Train.get_sentences_opinions()
    Train.get_aspects_polarities()
    Train.remove_elongate_chars()
    Train.remove_leading_spaces()
    Train.convert_lower_case()
    # Train.replace_char_in_list()
    Train.remove_special_chars()
    Train.tokenize()
    # Train.add_pos_neg_features()
    # Train.get_no_diacritic_sentences()
    # Train.preprocess()
    # Train.no_preprocess()

    train = pd.DataFrame({
        "Sentence": Train.sentences,
        "Aspect": Train.aspects,
        "Polarity": Train.polarities
    })

    train.to_csv("data/preprocessed_training_set.csv",
                 encoding="utf-8", index=False)

    print("[INFO] Prepare Dev Set...")
    Dev = FeedBack(dev_path)
    Dev.get_sentences_opinions()
    Dev.get_aspects_polarities()
    Dev.remove_elongate_chars()
    Dev.remove_leading_spaces()
    Dev.convert_lower_case()
    # Dev.replace_char_in_list()
    Dev.remove_special_chars()
    Dev.tokenize()
    # Dev.add_pos_neg_features()
    # Dev.get_no_diacritic_sentences()
    # Dev.preprocess()
    # Dev.no_preprocess()

    dev = pd.DataFrame({
        "Sentence": Dev.sentences,
        "Aspect": Dev.aspects,
        "Polarity": Dev.polarities
    })

    dev.to_csv("data/preprocessed_dev_set.csv",
               encoding="utf-8", index=False)

    print("[INFO] Prepare Test Set...")
    Test = FeedBack(test_path)
    Test.get_sentences_opinions()
    Test.get_aspects_polarities()
    Test.remove_elongate_chars()
    Test.remove_leading_spaces()
    Test.convert_lower_case()
    # Test.replace_char_in_list()
    Test.remove_special_chars()
    Test.tokenize()
    # Test.add_pos_neg_features()
    # Test.get_no_diacritic_sentences()
    # Test.preprocess()
    # Test.no_preprocess()

    test = pd.DataFrame({
        "Sentence": Test.sentences,
        "Aspect": Test.aspects,
        "Polarity": Test.polarities
    })

    test.to_csv("data/preprocessed_test_set.csv",
                encoding="utf-8", index=False)
