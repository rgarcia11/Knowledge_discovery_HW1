"""
This script tokenizes the input data and answers the excercise's questions
This exercise solution is presented by Rogelio Garcia.
"""
import os
import re
from heapq import nlargest
from nltk.stem import PorterStemmer

class FolderTokenizer:
    """
    This class has all the functions necessary to tokenize a folder of files.
    Inputs:
        pathToFiles - string containing the route to the folder that contains the files to tokenize
        pathToStopWords - if there is a stopwords catalogue to be used, it will eliminate stop words from the tokenized output
    This class builds a vocabulary, calculates total word count and word count per document
    Methods used:
        *Total word count checks if each encountered token is new and either adds it
        or sums one to the existing value, regardless of the document it was present in
        *Number of words sums one for each token encountered, disregarding if its a new token or not
        *Vocabulary builds an inversed index matrix with a hash table in the form
        of a Python Dictionary. Each entry in the table has another hash table with
        each document where the token was present in and the frequency inside that document
    """
    def __init__(self, pathToFiles, pathToStopWords=None, eliminateStopWords=False, stemmer=None):
        """
        Initializes the class with parameters
            pathToFiles - string containing the route to the folder that contains the files to tokenize
            pathToStopWords - if there is a stopwords catalogue to be used, it will eliminate stop words from the tokenized output
        The stop words are stored in a list called stopWords, and it expects the stop words catalogue in the format:
            word1
            word2
            .
            .
            .
            wordn
        with each word on a single line.
        """
        self.pathToFiles = pathToFiles
        self.vocabulary = {}
        self.numberOfWords = 0
        self.wordCount = {}
        self.stopWords = []
        self.stemmer = stemmer
        self.eliminateStopWords = eliminateStopWords
        if pathToStopWords:
            with open('./{}'.format(pathToStopWords)) as stopWordsFile:
                self.stopWords = [line.rstrip('\n') for line in stopWordsFile]
        self.iterateThroughFiles()

    def iterateThroughFiles(self):
        """
        This function iterates through every file in the folder and calls the
        tokenizer function for each file in the folder.
        """
        for filename in os.listdir(self.pathToFiles):
            with open('./{}/{}'.format(self.pathToFiles, filename)) as currentFile:
                self.tokenizer(currentFile, filename)

    def tokenizer(self, currentFile, filename):
        """
        This function receives a file and adds each token identified to the vocabulary.
        It removes punctuation, various special characters and numbers, and it
        separates tokens on spaces.
        """
        lines = [line.rstrip('\n') for line in currentFile]
        for line in lines:
            line = re.sub(r'[!@#%$^&*\(\)+0\\123456789:=\'.—/,\[\]\-\-_|\"?<>~]','',line)
            line = re.sub("'",'',line)
            afterSplit = line.strip().split(' ')
            for tokenCandidate in afterSplit:
                if tokenCandidate == ' ' or tokenCandidate == '':
                    continue
                self.numberOfWords += 1
                if self.eliminateStopWords and self.checkStopWord(tokenCandidate.lower()):
                    continue
                if self.stemmer:
                    tokenCandidate = self.stemmer.stem(tokenCandidate.lower())
                if tokenCandidate.lower() not in self.wordCount:
                    self.wordCount[tokenCandidate.lower()] = 1
                else:
                    self.wordCount[tokenCandidate.lower()] += 1

                if tokenCandidate.lower() not in self.vocabulary:
                    self.vocabulary[tokenCandidate.lower()] = {filename:1}
                else:
                    if filename not in self.vocabulary[tokenCandidate.lower()]:
                        self.vocabulary[tokenCandidate.lower()][filename] = 1
                    else:
                        self.vocabulary[tokenCandidate.lower()][filename] += 1

    def topWords(self, topNumber):
        """
        Returns the n most frequent words.
        Receives topNumber as parameter, the top n words wanted.
        Uses heapq library's nlargest function.
        """
        return nlargest(topNumber, self.wordCount, key=self.wordCount.get)

    def checkStopWord(self,word):
        """
        If there is a stop word catalogue, it checks if a given word is a stop word.
        It receives the word as parameters and outputs True or False.
        """
        if not self.stopWords:
            return False
        else:
            if word in self.stopWords:
                return True
            else:
                return False

    def sparseVector(self):
        words = list(self.vocabulary.keys())
        sparseVector = {}
        currentWord = 0
        for word in words:
            currentWord += 1
            files = self.vocabulary[word]
            for file in files:
                if file not in sparseVector:
                    sparseVector[file] = {currentWord:self.vocabulary[word][file]}
                else:
                    sparseVector[file][currentWord] = self.vocabulary[word][file]
        return sparseVector

if __name__ == '__main__':
    print('1. The class FolderTokenizer will tokenize the entire ./citeseer folder on whitespaces and removing punctuation, numbers and special characters.')
    folderTokenizer = FolderTokenizer('./citeseer','stopwords.txt')
    print('2. The following lines answer the questions to point 2.')
    print('a. There is a total of {} words in the collection.'.format(folderTokenizer.numberOfWords))
    print('b. The vocabulary size or number of unique terms is given by the size of the hash table: {}'.format(len(folderTokenizer.vocabulary)))
    topN = 20
    print('c. The top {} words by frequency are:'.format(topN))
    topWords = folderTokenizer.topWords(topN)
    stopWords = []
    for word in topWords:
        print(' {: >15} — {} appearances'.format(word, folderTokenizer.wordCount[word]))
        if folderTokenizer.checkStopWord(word):
            stopWords.append(word)
    print('d. Out of the list above, {} out of {} are stopwords.'.format(len(stopWords),len(topWords)))
    for word in stopWords:
        print(' {: >15}'.format(word))
    fifteenPercent = 0
    percentage = 0.0
    while percentage < 0.15:
        fifteenPercent += 1
        topFifteen = folderTokenizer.topWords(fifteenPercent)
        percentage += folderTokenizer.wordCount[topFifteen[len(topFifteen)-1]]/folderTokenizer.numberOfWords
    print('e. The minimum number of unique words accounting for 15% of the total number of words is {}'.format(fifteenPercent))

    print('3. The following lines answer the questions to point 3 using the PorterStemmer implementation of the NLKT Library.')
    folderTokenizerWithStemmer = FolderTokenizer('./citeseer','stopwords.txt', True, PorterStemmer())
    print('a. There is a total of {} words in the collection.'.format(folderTokenizerWithStemmer.numberOfWords))
    print('b. The vocabulary size or number of unique terms is given by the size of the hash table: {}'.format(len(folderTokenizerWithStemmer.vocabulary)))
    topN = 20
    print('c. The top {} words by frequency are:'.format(topN))
    topWords = folderTokenizerWithStemmer.topWords(topN)
    stopWords = []
    for word in topWords:
        print(' {: >15} — {} appearances'.format(word, folderTokenizerWithStemmer.wordCount[word]))
        if folderTokenizerWithStemmer.checkStopWord(word):
            stopWords.append(word)
    print('d. Out of the list above, {} out of {} are stopwords.'.format(len(stopWords),len(topWords)))
    for word in stopWords:
        print(' {: >15}'.format(word))
    fifteenPercent = 0
    percentage = 0.0
    while percentage < 0.15:
        fifteenPercent += 1
        topFifteen = folderTokenizerWithStemmer.topWords(fifteenPercent)
        percentage += folderTokenizerWithStemmer.wordCount[topFifteen[len(topFifteen)-1]]/folderTokenizerWithStemmer.numberOfWords

    print('e. The minimum number of unique words accounting for 15% of the total number of words is {}'.format(fifteenPercent))
    print('4. For this exercise we consider the dictionary (hash table) as a table, and calculate the transpose.')
    print('As the number of documents is probably quite large, only a couple examples will be printed.')
    sparseVector = folderTokenizerWithStemmer.sparseVector()
    sparseVectorKeys = list(sparseVector.keys())
    print(sparseVector[sparseVectorKeys[0]])
    print(sparseVector[sparseVectorKeys[1]])
