import re

import nltk
import pandas as pd
from nltk import SnowballStemmer
from nltk.corpus import stopwords

nltk.download('punkt')
nltk.download('stopwords')

stemmer = SnowballStemmer("german")
stop_words = set(stopwords.words("german"))


def clean_df(data: pd.DataFrame) -> pd.DataFrame:
    preprocessed_texts = []

    for text in data.text:
        text = clean(text)
        preprocessed_texts.append(text)

    data["clean"] = preprocessed_texts
    return data


def clean(text: str) -> str:
    """
        - remove any html tags (< /br> often found)
        - Keep only ASCII + European Chars and whitespace, no digits
        - remove single letter chars
        - convert all whitespaces (tabs etc.) to single wspace
        - all lowercase
        - remove stopwords, punctuation and stemm
    """
    text = text.lower()

    RE_WSPACE = re.compile(r"\s+", re.IGNORECASE)
    RE_TAGS = re.compile(r"<[^>]+>")
    RE_ASCII = re.compile(r"[^A-Za-zÀ-ž ]", re.IGNORECASE)
    RE_SINGLECHAR = re.compile(r"\b[A-Za-zÀ-ž]\b", re.IGNORECASE)

    text = re.sub(RE_TAGS, " ", text)
    text = re.sub(RE_ASCII, " ", text)
    text = re.sub(RE_SINGLECHAR, " ", text)
    text = re.sub(RE_WSPACE, " ", text)

    text = stemmer.stem(text)  # takes care of umlauts https://snowballstem.org/algorithms/german2/stemmer.html

    return text

