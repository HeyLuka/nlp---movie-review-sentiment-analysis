import os
import re
import nltk
from nltk.corpus import stopwords
from nltk.corpus import sentiwordnet as swn

def conductSynset(word, query):
	try:
		senti = swn.senti_synset(query)
		# return (word, senti.pos_score()-senti.neg_score())
		return senti.pos_score()-senti.neg_score()
	except:
		return 0

def extractLabel(fn):
	fn_split = fn.split('.')
	fn_split_2 = fn_split[0].split('_')
	rating = fn_split_2[1]
	if int(rating)>=7:
		return 1 # positive review
	else:
		return 0 # negative review

def getSWN(tokens_tags):
	posREList = [re.compile(item) for item in ['NN','VB','JJ','RB'] ]
	# print(posREList)

	swnList = []

	# filter pos and pick out those we care
	for each in tokens_tags:
		w = each[0]
		pos = each[1]
		if posREList[0].match(pos)!=None:
			# pos - n
			comb = w + '.n.01'
			# res = swn.senti_synset(comb)
			res = conductSynset(w, comb)
			swnList.append(res)
		elif posREList[1].match(pos)!=None:
			# pos - v
			comb = w + '.v.01'
			# res = swn.senti_synset(comb)
			res = conductSynset(w, comb)
			swnList.append(res)
		elif posREList[2].match(pos)!=None:
			# pos - adj
			comb = w + '.a.01'
			# res = swn.senti_synset(comb)
			res = conductSynset(w, comb)
			swnList.append(res)
		elif posREList[3].match(pos)!=None:
			# pos - adv
			comb = w + '.r.01'
			# res = swn.senti_synset(comb)
			res = conductSynset(w, comb)
			swnList.append(res)
		else:
			continue
	return swnList

sentiList = []
labelList = []

for filename in os.listdir('../dataset'):
	with open('../dataset/'+filename, 'r') as fr:
		review = fr.read()
		lower_review = review.lower()
		tokens = nltk.regexp_tokenize(lower_review, r'\w+')
		filtered_tokens = [word for word in tokens if word not in stopwords.words('english')]

		tokensTags = nltk.pos_tag(filtered_tokens)

		reviewSWN = getSWN(tokensTags)
		sentiList.append((filename, sorted(reviewSWN, key=abs, reverse=True)[0:10]))
	labelList.append(extractLabel(filename)) 

print(sentiList, end='\n')
print(labelList)
