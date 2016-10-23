import os
import nltk
from nltk.corpus import stopwords
from nltk.corpus import sentiwordnet as swn

with open('12456_10.txt', 'r') as fr:
	review = fr.read()

lower_review = review.lower()
# tokens = nltk.word_tokenize(lower_review)
tokens = nltk.regexp_tokenize(lower_review, r'\w+')
filtered_tokens = [word for word in tokens if word not in stopwords.words('english')]
# print(lower_review)
# print(tokens)
# print(filtered_tokens)
print(nltk.pos_tag(filtered_tokens))
