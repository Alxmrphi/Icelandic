## Orð2Vec - Quick word embedding tutorial for Icelandic

This tutorial aims to illustrate how to build a simple word embedding model for Icelandic using Word2Vec, implemented in the gensim library. Many of the details will be overlooked as the goal is simply to show how to create your own model and some of the interesting applications that can be applied.

### Prerequisites

To run the code used in this article it is recommended that you have [Anaconda](https://www.continuum.io/downloads) installed as well as [gensim](https://radimrehurek.com/gensim/install.html).

### Word embedding models

The idea behind word embedding is to look at a word in its surrounding context and derive a representation for it that exists in a high dimensional vector space. Even if the words 'dog' and 'canine' do not ever occur together in a specific training corpus, the environments in which they do occur (dictated by the surrounding context words) are enough for their embedding to be similar. Then, mathemetical measures of similarity on vectors would be able to detect that 'canine' is in fact a very similar word to 'dog' and less so to the word 'apple'. This idea is captured nicely in what is known as the distributional hypothesis.

### The distributional hypothesis

The distributional hypothesis is stated in many closely-related forms. I hope by just picking a few quotes from other sources that the idea becomes clear:

(1) You shall know a word by the company it keeps

(2) Words that occur in similar contexts have similar representations

(3) The meaning of a word can be understood from the words that tend to occur around it.



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
### size

The size parameter dictates the dimensionality of each word embedding. Word2Vec's default is 100, but many large applications of Word2Vec use significantly more than this. The larger the dimensionality, the more data needed to train the model in order for those feature vectors to be _good_. There is a fair amount of data in the MÍM corpus, so I chose 250 features in each word embedding. Word2Vec uses a neural network with only a single hidden layer, with **"size"** nodes in the hidden layer. The learned weights of each neuron in the hidden layer after training are exactly these features. Therefore, the larger the dimensionality of your word vectors, the more nodes are in the hidden layer. This is why increasing the dimensionality requires more data to learn good representations. The more features any model has, the more data is required to fit them.

### window

Given the definitions of the distributional hypothesis above, one might ask how close does a word have to occur with another word for it to be considered relevant to its vector space representation. This is defined by the **window** parameter. Word2Vec's default is 5, which means as the model iteratively runs through the corpus, only words a distance of maximum **window** words away are considered. This means that if a context word always occurs six words before / after a target word during training, the model will fail to pick this up. Increasing the window size allows for more general inference of words that are more loosely connected to the specific target word being trained. Shorter windows are useful for determining words that are functionally similar, synonyms etc. The default window size (5) is used in this implementation.

### min_count

This parameter sets the threshold for words we consider as part of the vocabulary. Very rare words, that only occur once or twice in our corpus, are pretty useless given that there isn't enough data to learn a good word embedding for them. Therefore, by purging these words from the vocabulary, it makes the training phase more efficient and allows us to fit more common words in our window of interest (described above), which greatly aids the modelling process.

Now, we are ready to train the model:

```markdown
model = Word2Vec(corpus, size=250, window=5, min_count=3)
```
## Inspecting the model

Now that the model has been trained, we can visualise a raw vector to see what one looks like

We can also compare the similarity between two word vectors:



## Evaluation

```markdown
In: model.most_similar(positive=['kona', 'kóngur'], negative=['maður'], topn=1)
Out: [('drottning', 0.6090534925460815)]

In: model.most_similar(positive=['kona', 'drengur'], negative=['maður'], topn=1)
Out: [('stúlka', 0.7293691039085388)]

In: model.most_similar(positive=['hann', 'kvenkyn'], negative=['hún'], topn=1)
Out: [('karlkyn', 0.7098826169967651)]

In: model.most_similar(positive=['hundur', 'kettlingur'], negative=['köttur'], topn=1)
Out: [('hvolpur', 0.7297102212905884)]

In: model.most_similar(positive=['skóli', 'prófessor'], negative=['háskóli'], topn=1)
Out: [('skólastjóri', 0.5468635559082031)]
```
But it is not always completely correct. Consider the following, where an expected word would be _höfn_:
```markdown
In: model.most_similar(positive=['bátur', 'flugvöllur'], negative=['flugvél'], topn=1)
Out: [('lóð', 0.5703256130218506)]
```
It looks like the corpus (with our predefined hyperparameters) doesn't allow for such a good inference in this case.
However, perhaps given that airports are more represented in the corpus rather than harbours, it could pick up on the correct semantic relation with a different ordering of the word vectors:
```markdown
In: model.most_similar(positive=['flugvél', 'höfn'], negative=['bátur'], topn=1)
Out: [('flugvöllur', 0.6099342107772827)]
```
As expected!

It can even model the relationship among grammatical terms, from a corpus that doesn't appear to have any explicit grammatical descriptions (please correct me if I am wrong). For example, what is, to a noun, what an adverb is to a verb?
```markdown
model.most_similar(positive=['nafnorð', 'atviksorð'], negative=['sagnorð'], topn=1)
Out: [('lýsingarorð', 0.6917917132377625)]
```
An adjective!

The word embeddings shown here are all good examples of good modelling, but you shouldn't think this is a perfect model by any means. For example, here are closest word vectors to _máltækni_ 

```markdown
In: model.most_similar('máltækni')
Out[
[('wallau-massenheim', 0.8737115859985352),
 ('paprikuduft', 0.8696819543838501),
 ('categories', 0.8686853647232056),
 ('31:32', 0.8685649037361145),
 ('15:51', 0.8675534129142761),
 ('08:37', 0.8662518858909607),
 ('14:01', 0.8656798005104065),
 ('koparendur', 0.8647289276123047),
 ('stjórnunarráðgjafi', 0.8633975386619568),
 ('flatvagnar', 0.8632863163948059)]
```
Perhaps it is too cruel to judge such a small corpus on how well it can model such a technical word, even if its closest semantic neighbour is a German handball team, closely followed by paprika powder.

Here are the most similar word vectors for (1) **pilot**, (2) **apple** and (3) **Iceland**:

```markdown
model.most_similar('flugmaður')
Out: 
[('flugstjóri', 0.6587222218513489),
 ('farþegi', 0.6484204530715942),
 ('flugvél', 0.6159329414367676),
 ('þota', 0.6112388372421265),
 ('þyrlur', 0.6025561094284058),
 ('varðskip', 0.5978344082832336),
 ('áhöfn', 0.592880368232727),
 ('bílstjóri', 0.5810525417327881),
 ('þyrla', 0.5693312883377075),
 ('herskip', 0.5687568783760071)]
 
In: model.most_similar('epli')
Out: 
[('baun', 0.9056194424629211),
 ('gulrót', 0.8846516609191895),
 ('hneta', 0.880598783493042),
 ('vínber', 0.8719343543052673),
 ('paprika', 0.8683988451957703),
 ('appelsína', 0.8601030111312866),
 ('ólífa', 0.8554715514183044),
 ('ostur', 0.8518120050430298),
 ('sulta', 0.8465861082077026),
 ('salat', 0.8437827825546265)]
 
model.most_similar('ísland')
Out: 
[('íslendingur', 0.5155368447303772),
 ('norðurland', 0.4860343039035797),
 ('land', 0.4806867241859436),
 ('hérlendis', 0.4788222908973694),
 ('danmörk', 0.4710685908794403),
 ('grænland', 0.4706999659538269),
 ('noregur', 0.4664274752140045),
 ('akureyri', 0.4584369957447052),
 ('bifröst', 0.43473801016807556),
 ('írlandi', 0.4321120083332062)]
 
```
These seem more sensible. This goes to show the effect that good qualitiative data has on word embedding models (or rather, any model).


