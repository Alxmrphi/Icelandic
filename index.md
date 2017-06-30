## Orð2Vec - Quick word embedding tutorial for Icelandic

This tutorial aims to illustrate how to build a simple word embedding model for Icelandic using Word2Vec, implemented in the gensim library. Many of the details will be overlooked as the goal is simply to show how to create your own model and some of the interesting applications that can be applied.

### Word embedding models

The idea behind word embedding is to look at a word in its surrounding context and derive a representation for it that exists in a high dimensional vector space. Even if the words 'dog' and 'canine' do not ever occur together in a specific training corpus, the environments in which they do occur (dictated by the surrounding context words) are enough for their embedding to be similar. Then, mathemetical measures of similarity on vectors would be able to detect that 'canine' is in fact a very similar word to 'dog' and less so to the word 'apple'. This idea is captured nicely in what is known as the distributional hypothesis.

### The distributional hypothesis

The distributional hypothesis is stated in many closely-related forms. I hope by just picking a few quotes from other sources that the idea becomes clear:

(1) You shall know a word by the company it keeps
(2) Words that occur in similar contexts have similar representations
(3) The meaning of a word can be understood from the words that tend to occur around it



### Skip-gram vs CBOW

### Dataset

The corpus used to generate the word embeddings is _Mörkuð Íslensk Málheild_ and can be found at [malfong.is](http://malfong.is/?pg=mim). It is a corpus of 499,355 sentences with 165,081 lemmas that appear at least three times. After downloading and extracting the data to a new folder, you can define the following python class that will read the folder iteratively while the model is trained. This means that you do not need to store the entire corpus (1.7GB) in RAM at the same time (very handy).

```markdown
import os
from xml.dom.minidom import parse

class MIM_Parser(object):
    def __init__(self, mim_folder):
        self.mim_folder = mim_folder
 
    def __iter__(self):
        for folder in os.listdir(self.mim_folder):
            if os.path.isdir(folder):
                current_folder = os.path.join(self.mim_folder, folder)
                for file in os.listdir(current_folder):
                    root = parse(os.path.join(current_folder, file))
                    for sentence in root.getElementsByTagName('s'):
                        words = sentence.getElementsByTagName('w')
                        cs = [] # current sentence
                        for word in words:
                            cs.append(word.getAttribute('lemma'))
                        yield cs
```
Now you can instantiate the class and point it to your the new directly where the text folders are:

```markdown
corpus = MIM_Parser('/home/username/MIM_data')
```
This implementation only selects the lemmas as I doubt there is enough data to learn a good representation for every possible form of a word across all cases and number. A lot of information would be lost if each word declension is treated as a separate vocabulary item. Perhaps easier access to larger data sets in the future might change that. This has an implication on the kinds of questions we can ask the model, namely that we cannot inspect vector dimensions that encode for case or tense when only modelling the lemmas.

Before building the model, there is a bit to say about some of the parameters that need to be chosen.
#### size

The size parameter dictates the dimensionality of each word embedding. Word2Vec's default is 100, but many large applications of Word2Vec use significantly more than this. The larger the dimensionality, the more data needed to train the model in order for those feature vectors to be _good_. There is a fair amount of data in the MÍM corpus, so I chose 250 features in each word embedding. Word2Vec uses a neural network with only a single hidden layer, with **"size"** nodes in the hidden layer. The learned weights of each neuron in the hidden layer after training are exactly these features. Therefore, the larger the dimensionality of your word vectors, the more nodes are in the hidden layer. This is why increasing the dimensionality requires more data to learn good representations from.

Markdown is a lightweight and easy-to-use syntax for styling your writing. It includes conventions for

```markdown
Syntax highlighted code block

# Header 1
## Header 2
### Header 3

- Bulleted
- List

1. Numbered
2. List

**Bold** and _Italic_ and `Code` text

[Link](url) and ![Image](src)
```

For more details see [GitHub Flavored Markdown](https://guides.github.com/features/mastering-markdown/).

### Jekyll Themes

Your Pages site will use the layout and styles from the Jekyll theme you have selected in your [repository settings](https://github.com/Alex159/Icelandic/settings). The name of this theme is saved in the Jekyll `_config.yml` configuration file.

### Support or Contact

Having trouble with Pages? Check out our [documentation](https://help.github.com/categories/github-pages-basics/) or [contact support](https://github.com/contact) and we’ll help you sort it out.
