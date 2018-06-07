"""
This script tokenizes the input data and answers the excercise's questions
This exercise solution is presented by Rogelio Garcia.
"""
import os
import re
from heapq import nlargest

class FileHandler:
    def __init__(self, pathToFiles, pathToStopWords=None):
        self.pathToFiles = pathToFiles
        self.vocabulary = {}
        self.numberOfWords = 0
        self.wordCount = {}
        with open('./{}'.format(pathToStopWords)) as stopWordsFile:
            self.stopWords = [line.rstrip('\n') for line in stopWordsFile]

    def iterateThroughFiles(self):
        currentFileNumber = 0
        for filename in os.listdir(self.pathToFiles):
            with open('./{}/{}'.format(self.pathToFiles, filename)) as currentFile:
                self.tokenizer(currentFile, currentFileNumber)
            currentFileNumber = currentFileNumber + 1

    def tokenizer(self, currentFile, currentFileNumber):
        lines = [line.rstrip('\n') for line in currentFile]
        for line in lines:
            line = re.sub(r'[!@#%$^&*\(\)+0\\123456789:=\'.—/,\[\]\-\-_|\"?<>~]','',line)
            line = re.sub("'",'',line)
            afterSplit = line.strip().split(' ')
            for tokenCandidate in afterSplit:
                if tokenCandidate == ' ' or tokenCandidate == '':
                    continue
                self.numberOfWords += 1
                if tokenCandidate.lower() not in self.wordCount:
                    self.wordCount[tokenCandidate.lower()] = 1
                else:
                    self.wordCount[tokenCandidate.lower()] += 1

                if tokenCandidate.lower() not in self.vocabulary:
                    self.vocabulary[tokenCandidate.lower()] = {currentFileNumber:1}
                else:
                    if currentFileNumber not in self.vocabulary[tokenCandidate.lower()]:
                        self.vocabulary[tokenCandidate.lower()][currentFileNumber] = 1
                    else:
                        self.vocabulary[tokenCandidate.lower()][currentFileNumber] += 1

    def topWords(self, topNumber):
            return nlargest(topNumber, self.wordCount, key=self.wordCount.get)

    def checkStopWord(self,word):
        if not self.stopWords:
            return False
        else:
            if word in self.stopWords:
                return True
            else:
                return False

if __name__ == '__main__':
    fileHandler = FileHandler('./citeseer','stopwords.txt')
    fileHandler.iterateThroughFiles()
    print('The following lines answer the questions to point 2.')
    print('a. There is a total of {} words in the collection.'.format(fileHandler.numberOfWords))
    print('b. The vocabulary size or number of unique terms is given by the size of the hash table: {}'.format(len(fileHandler.vocabulary)))
    print('c. The top 20 words by frequency are:')
    topTwentyWords = fileHandler.topWords(20)
    fifteenPercent = 0
    percentage = 0.0
    stopWords = []
    for word in topTwentyWords:
        print('{} — {} appearances'.format(word, fileHandler.wordCount[word]))
        if percentage < 0.15:
            percentage += fileHandler.wordCount[word]/fileHandler.numberOfWords
            fifteenPercent += 1
        if fileHandler.checkStopWord(word):
            stopWords.append(word)

    print('d. Out of the list above, {} out of {} are stopwords:'.format(len(stopWords),len(topTwentyWords)))
    for word in stopWords:
        print(word)
    print('e. The minimum number of unique words accounting for 15% of the total number of words is {}'.format(fifteenPercent))
