from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import regex


#### LATEX CLEANING UTILITIES


## 1. Latin-ize latex accents enclosed in brackets
def remove_latex_accents(string):
    accent = r"\\[\'\"\^\`H\~ckl=bdruvtoi]\{([a-z])\}"
    replacement = r"\1"

    string = regex.sub(accent, replacement, string)
    return string


## 2. Remove latex environments
def remove_env(string):
    env = r"\\[a-z]{2,}{[^{}]+?}"

    string = regex.sub(env, "", string)
    return string


## 3. Latin-ize non-{} enclosed latex accents:
def remove_accents(string):
    accent = r"\\[\'\"\^\`H\~ckl=bdruvtoi]([a-z])"
    replacement = r"\1"

    string = regex.sub(accent, replacement, string)
    return string


## 4. ONLY remove latex'd math that is separated as a 'word' i.e. has space characters on either side of it.


def remove_latex(string):
    latex = r"\s(\$\$?)[^\$]*?\1\S*"
    string = regex.sub(latex, " LATEX ", string)
    return string


def cleanse(string):
    string = string.replace("\n", " ")
    string = remove_latex_accents(string)
    string = remove_env(string)
    string = remove_accents(string)
    string = remove_latex(string)
    return string
