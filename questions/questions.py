import math
import nltk
import os
import string
import sys

FILE_MATCHES = 1
SENTENCE_MATCHES = 1


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python questions.py corpus")

    # Calculate IDF values across files
    files = load_files(sys.argv[1])
    file_words = {
        filename: tokenize(files[filename])
        for filename in files
    }
    file_idfs = compute_idfs(file_words)

    # Prompt user for query
    query = set(tokenize(input("Query: ")))

    # Determine top file matches according to TF-IDF
    filenames = top_files(query, file_words, file_idfs, n=FILE_MATCHES)

    # Extract sentences from top files
    sentences = dict()
    for filename in filenames:
        for passage in files[filename].split("\n"):
            for sentence in nltk.sent_tokenize(passage):
                tokens = tokenize(sentence)
                if tokens:
                    sentences[sentence] = tokens

    # Compute IDF values across sentences
    idfs = compute_idfs(sentences)

    # Determine top sentence matches
    matches = top_sentences(query, sentences, idfs, n=SENTENCE_MATCHES)
    for match in matches:
        print(match)


def load_files(directory):
    """
    Given a directory name, return a dictionary mapping the filename of each
    `.txt` file inside that directory to the file's contents as a string.
    """
    contents = dict()
    for filename in os.listdir(directory):
        with open(os.path.join(directory, filename), "r") as file:
            contents[filename] = file.read()
    return contents


def tokenize(document):
    """
    Given a document (represented as a string), return a list of all of the
    words in that document, in order.

    Process document by coverting all words to lowercase, and removing any
    punctuation or English stopwords.
    """
    words = nltk.word_tokenize(document.lower())
    words = [word for word in words if word not in nltk.corpus.stopwords.words("english")]
    words = [word for word in words if not all(char in string.punctuation for char in word)]
    return words


def compute_idfs(documents):
    """
    Given a dictionary of `documents` that maps names of documents to a list
    of words, return a dictionary that maps words to their IDF values.

    Any word that appears in at least one of the documents should be in the
    resulting dictionary.
    """
    counts = dict()
    for filename in documents:
        seen = set()
        for word in documents[filename]:
            if word not in seen:
                counts[word] = counts.get(word, 0) + 1
                seen.add(word)
    idf = {word: math.log(len(documents) / counts[word]) for word in counts}
    return idf


def top_files(query, files, idfs, n):
    """
    Given a `query` (a set of words), `files` (a dictionary mapping names of
    files to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the filenames of the the `n` top
    files that match the query, ranked according to tf-idf.
    """
    tf_idf = dict()
    for filename in files:
        for word in query:
            tf_idf[filename] = tf_idf.get(filename, 0) + files[filename].count(word) * idfs[word]
    results = sorted(tf_idf.items(), key = lambda x: x[1], reverse = True)[:n]
    results = [result[0] for result in results]
    return results


def top_sentences(query, sentences, idfs, n):
    """
    Given a `query` (a set of words), `sentences` (a dictionary mapping
    sentences to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the `n` top sentences that match
    the query, ranked according to idf. If there are ties, preference should
    be given to sentences that have a higher query term density.
    """
    info = dict()
    for sentence in sentences:
        info[sentence] = [0, 0] # idf, qtd
        for word in query:
            if word in sentences[sentence]:
                info[sentence][0] += idfs[word]
                info[sentence][1] += sentences[sentence].count(word) / len(sentences[sentence])
    results = sorted(info.items(), key = lambda x: (x[1][0], x[1][1]), reverse = True)[:n]
    results = [result[0] for result in results]
    return results

if __name__ == "__main__":
    main()
