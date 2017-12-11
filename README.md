# Authorship-Identification
Text analysis in NLP to identify author of piece of text

We aim to do authorship identification on lines of poetry written by Emily Bronte and William Shakespeare. We’ll be using the Naive Bayes classifier provided by NLTK. We try to tune in feeatures to train the model as perfectly as possible.

Data given:
<li>  s.data Lines of poetry by Shakespeare, marked with an ’s’ </li>
<li>  b.data Lines of poetry by Bronte, marked with an ’b’ </li>

## preppring data into train and test:
cat s.data b.data | python maketsv.py -o train.tsv
python classify.py -i train.tsv

Features added to the classifier:
--------------------
1.1 <strong>Morphological form of words</strong> present in the CMU dict in every sentence added with weight 1.
1.2 If word does not exist in CMU dict and is not a punctuation, added the word with a weight 2, as it could be a representative of<strong> old style of writing</strong>.

2. <strong>Length of line :</strong> As researched from a few online resources,it was quite evident that the number of words in a line for Shakespeare poetry are greater than number of words in a line for Emily Bronte.

3. <strong>Number of Syllables:</strong> Shakespeare's sonnets are essentially written in a pattern, that has 10 syllables or more. 

4. <strong>Stress on Word:</strong> Shakespeare's sonnets follow a stress pattern of '1010101010' , which may be inversed at times. Such a pattern may be observable in few lines of Emily Bronte, but it is not a pure writing style for her.

5. <strong>Iambic Pentameter:</strong> While Emily Bronte's poetry is more lyrical i.e. following a rhyming pattern, Shakespeare's poetry is essentially written following Iambic Pentameter. The term Iambic Pentameter describes the rhythm that the words establish in that line, which is measured in small groups of syllables called "feet".  The word "iambic" refers to the type of foot that is used, known as the iamb, which in English is an unstressed syllable followed by a stressed syllable. 
The word "pentameter" indicates that a line has five of these "feet".
https://en.wikipedia.org/wiki/Iambic_pentameter

 Blank verse vs Prose : Shakespeare poetry is written in verse format i.e. rhythmical, whereas, Emily followed Prose style of writing.
 
6. <strong>Punc_normalized words:</strong> Words obtained after removing punctuations and normalizing other morphological forms. eg. makest --> make


Additional Approaches tried:
----
1. Adding poetry of authors, whose writing style influenced Shakespeare and followed Iambic Pentameter, to say with confidence if a stress pattern was same as the one in external file, it is a Shakespearen writing.
But, due to many old style words and huge number of stress patterns, it could not be used with much success.

2. Adding the dictionary of old words as an nltk library : to find the roots / morphenes of poetically modified words, did not make change to the accuracy.

3. Checking for rhyming : As Emily Bronte's poems follow "ABCB" rhyming pattern, it could have been an essential feature in distinguishing the two authors.
But, the given data does not contain consecutive lines to provide a proof of this observation.

4.Adding n-gram counts of b.data, s.data as an additional feature

Output:
---
Training classifier ... 
Accuracy on dev: 0.946794 
