'''Create a file called semantic.py and run all the code extracts above.
Write a note about what you found interesting about the similarities
between cat, monkey and banana and think of an example of your own.'''

import spacy
nlp = spacy.load('en_core_web_md')

tokens = nlp('cat apple monkey banana ')
for token1 in tokens:
    for token2 in tokens:
        print(token1.text, token2.text, token1.similarity(token2))

''' I find it interesting that the code can assess the similarity and knows that monkey and banana
are more similar as monkeys tend to like bananas. It is interesting how it can categorise words 
and know its meanings and common features. My example can be dog bone milk and lizard'''

tokens = nlp('dog bone milk lizard ')
for token1 in tokens:
    for token2 in tokens:
        print(token1.text, token2.text, token1.similarity(token2))

'''I like how it picked up that dogs may like bones and bone and milk have higher similarity. 
I guessing it's cause milk makes your bones healthier'''


sentence_to_compare = "Why is my cat on the car"
sentences = ["where did my dog go",
"Hello, there is my car",
"I\'ve lost my car in my car",
"I\'d like my boat back",
"I will name my dog Diana"]
model_sentence = nlp(sentence_to_compare)
for sentence in sentences:
    similarity = nlp(sentence).similarity(model_sentence)
    print(sentence + " - ", similarity)


# Run the example file with the simpler language model ‘en_core_web_sm’
# and write a note on what you notice is different from the model 'en_core_web_md'.

'''After runnung example.py in a different model I got this :UserWarning: [W007] The model you're using has no word vectors loaded, 
so the result of the Doc.similarity method will be based on the tagger, parser and NER, which may not give useful similarity judgements. 
This may happen if you're using one of the small models, e.g. `en_core_web_sm`, which don't ship with word vectors and only use
context-sensitive tensors. You can always add your own word vectors, or use one of the larger models instead if available.
Also the numbers were completely different.

After futher reseatch I found out that:
'Accuracy: The en_core_web_md model is more accurate than en_core_web_sm 
as it has been trained on a larger corpus of data, which means it can recognize
 and parse more complex sentence structures and subtle nuances in language.'

'Speed: Since the en_core_web_sm model is smaller in size, it is faster to load
and process compared to en_core_web_md. However, en_core_web_md is still relatively fast and efficient for most NLP tasks.' '''
