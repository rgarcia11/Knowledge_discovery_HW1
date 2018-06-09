# Knowledge_discovery_HW1
First homework of the Knowledge Discovery from Social and Information Networks summer class in Universidad de los Andes.
This work is presented by **Rogelio Garcia**. It is available at the course's *virtual machine* and at *Sicuaplus*.
It was solved in Python3.3 using two libraries, NLTK and heapq.
## Questions
### 1. Write a program that preprocesses the collection. In doing so, tokenize on whitespace and remove punctuation.
The solution described in the next section tokenizes on whitespaces, removes punctuation, numbers and special characters.
The characters removed are as follow:

> !@#%$^&*{}()+0123456789:='.:;--—\/,[]_|"?<>~

### 2. Determine the frequency of occurrence for all the words in the collection. Answer the following questions:
#### a. What is the total number of words in the collection?
The total number of words in the collection is 472077.
#### b. What is the vocabulary size? (i.e., number of unique terms).
The number of unique terms or vocabulary size is 19215.
#### c. What are the top 20 words in the ranking? (i.e., the words with the highest frequencies).
|    Word     | Appearances |
| ----------: | :---------- |
|         the | 25661       |
|          of | 18640       |
|         and | 14131       |
|           a | 13361       |
|          to | 11536       |
|          in | 10067       |
|         for |  7379       |
|          is |  6578       |
|          we |  5138       |
|        that |  4820       |
|        this |  4446       |
|         are |  3737       |
|          on |  3656       |
|          an |  3281       |
|        with |  3200       |
|          as |  3057       |
|          by |  2765       |
|        data |  2691       |
|          be |  2500       |
| information |  2321       |

#### d. From these top 20 words, which ones are stop-words?
18 out of the top 20 words are stop words, only data and information are not stopwords.

> Stopwords: the, of, and, a, to, in, for, is, we, that, this, are, on, an, with, as, by, be.

#### e. What is the minimum number of unique words accounting for 15% of the total number of words in the collection?
One would only need to add the frequencies of the first **4 words** to have over 15% of the total number of words.
|    Word     | Appearances |
| ----------: | :---------- |
|         the | 25661       |
|          of | 18640       |
|         and | 14131       |
|           a | 13361       |
|       total | 71793       |

> 71791/472077=15.2%
### 3. Integrate the Porter stemmer and a stopword eliminator into your code. Answer again questions a.-e. from the previous point.
The stopword catalogue was imported from https://www.dropbox.com/s/5789sj8v07j2id0/stopwords.txt.
The Porter stemmer was imported from the NLTK Python library. The details on how it was implemented are on the next section "Solution".
#### a. What is the total number of words in the collection?
The total number of words in the collection is, again, 472077, it should not variate.
#### b. What is the vocabulary size? (i.e., number of unique terms).
The number of unique terms or vocabulary size is 12886.
#### c. What are the top 20 words in the ranking? (i.e., the words with the highest frequencies).
|    Word   | Appearances |
| --------: | :---------- |
|    system | 3741        |
|      data | 2691        |
|     agent | 2686        |
|    inform | 2397        |
|     model | 2313        |
|     paper | 2246        |
|     queri | 1906        |
|      user | 1756        |
|     learn | 1740        |
| algorithm | 1584        |
|  approach | 1544        |
|   problem | 1543        |
|    applic | 1522        |
|   present | 1507        |
|      base | 1487        |
|       web | 1439        |
|   databas | 1424        |
|    comput | 1411        |
|    method | 1223        |
|    result | 1202        |

#### d. From these top 20 words, which ones are stop-words?
None of them are stopwords because we didn't add stopwords into the vocabulary.

#### e. What is the minimum number of unique words accounting for 15% of the total number of words in the collection?
The same exercise was done again for this question, but now **58 words** are needed to cover 15% of the total word count.
This is noticeably greater than the 4 words needed when stopwords were included.

### 4. Encode each document using the sparse TF-IDF representation.
The sparse TF-IDF notation was generated for each article. The output is a dictionary containing each document's name, and for each one another dictionary with the notation.
As the number of documents is probably quite large, I will show a couple examples.

<pre>

100157: {1: 6, 2: 6, 3: 7, 4: 7, 5: 2, 6: 2, 7: 1, 8: 1, 9: 1, 10: 4, 11: 1, 12: 1, 13: 1, 14: 1, 15: 5, 16: 1, 17: 1, 18: 2, 19: 4, 20: 1, 21: 1, 22: 2, 23: 1, 24: 1, 25: 1, 26: 2, 27: 1, 28: 1, 29: 1, 30: 1, 31: 2, 32: 1, 33: 1, 34: 1, 35: 1, 36: 1, 37: 1, 38: 1, 39: 1, 40: 1, 41: 1, 42: 1, 43: 1, 44: 1}

108573: {1: 1, 13: 1, 24: 1, 27: 1, 35: 2, 39: 2, 43: 1, 74: 3, 110: 1, 126: 1, 132: 3, 160: 2, 169: 1, 171: 1, 239: 1, 285: 1, 286: 7, 287: 2, 484: 1, 490: 1, 514: 1, 521: 1, 523: 1, 536: 6, 537: 4, 538: 4, 539: 2, 540: 1, 541: 1, 542: 1, 543: 1, 544: 3, 545: 1, 546: 1, 547: 1, 548: 1, 549: 1, 550: 1, 551: 1, 552: 1, 553: 2, 554: 1, 555: 1, 556: 1, 557: 1, 558: 1, 559: 2, 560: 1, 561: 1, 562: 1, 563: 1, 564: 1, 565: 1, 566: 1, 567: 1, 568: 1, 569: 1}

</pre>

## Solution
A single Python script was written to solve everything. It is comprised of two parts: A tokenizer class and a section that runs the each answer for the exercise. The information provided above is taken from what the program prints out. The tokenizer class reads each file in the folder and tokenizes it, it also maintains the vocabulary, generates the notation, removes stopwords... At first it seems like a single class is concentrating everything, but this class can be instantiated with or without removing stopwords, or without a stemmer. It can also use different stemmers as long as said stemmer has a "stem" method and takes a word as input.
The only "hardcoded" bit of the class is the removal of punctuation, numbers and special characters, but I found it simpler to write it in the code instead of using for example a file with the characters to remove due to string codification issues I ran into.

The code is provided and it is fully documented.

### Libraries and imports
The libraries used are:

| Library | Use                                                                   |
| ------: | :-------------------------------------------------------------------- |
|      os | Used to read and access files.                                        |
|      re | Used to manage regular expressions to delete special characters.      |
|   heapq | Utility library used to find the biggest N in a dictionary.           |
|    nltk | Natural language processing tool kit used to import a Porter stemmer. |

### Class FolderTokenizer
Again, it is fully documented, so I'll explain the functionality.
The constructor has the following parameters:

> def \_\_init\_\_(self, pathToFiles, pathToStopWords=None, eliminateStopWords=False, stemmer=None):

It may or may not receive a path to the stop words catalogue to be used, and it may or may not eliminate stop words. It also may or may not use a stemmer. So for the questions in point 2 the FolderTokenizer was instantiated like this:

> folderTokenizer = FolderTokenizer('./citeseer','stopwords.txt')

Because it was only needed to check which words were stopwords, and to tokenize everything without eliminating stopwords or stemming them (not that the class assumes False on the eliminateStopWords input).

For the second part, however, it was instantiated like this:

> folderTokenizerWithStemmer = FolderTokenizer('./citeseer','stopwords.txt', True, PorterStemmer())

Explicitly telling the class to eliminate stopwords and to use a stemmer. Note that this PorterStemmer() comes from the import *from nltk.stem import PorterStemmer*.

## How to run
In order to run this script, you will need Python3 and the libraries nltk, re and heapq. The virtual machine already contains them.
To run, step into the project folder and simply use the command:

> python3 tokenizer.py

*Note that depending on how you installed python, it might be only python tokenizer.py.*

## Limitations
Limitations include some word separations. In the case of a misread, eliminating special characters may end up in joining words. For example:
<pre>
??This is a Title??
And this is a paragraph
</pre>
When removing the characters *?* and newline character *\n*, the word *Title* and the word *And* will be joined, resulting in a token being titleand.

With joint terms like, for example, "machine-learning", it would result in a token being machinelearning.

With situations like "Adapt/create", the read token would be adaptcreate.
