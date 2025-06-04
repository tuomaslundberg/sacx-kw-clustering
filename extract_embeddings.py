import sys
import os
import torch
import datasets
import re
from string import punctuation
from transformers import AutoTokenizer, AutoModelForSequenceClassification

SPECIAL_TOKENS = {"<s>", "</s>", "<pad>", "<unk>"}

# To make accessing path arguments easier
arg_keys = ['data_dir', 'model_dir', 'result_dir', 'lang', 'fold']
args = dict(zip(arg_keys, sys.argv[1:]))

hf_path = os.path.join(args['data_dir'], f"data-with-keywords-{args['lang']}")
model_path = os.path.join(args['model_dir'], f"{args['fold']}")
emb_path = os.path.join(args['result_dir'], f"embeddings-{args['lang']}.csv")
os.makedirs(args['result_dir'], exist_ok=True)

def tokenize_fn(ex):
	if ex is None:
		return tokenizer("", return_tensors='pt', truncation=True, padding=True)
	return tokenizer(ex['text'], return_tensors='pt', truncation=True, padding=True)

def enrich_dataset_fn(ex):
	doc_tokens = tokenizer.convert_ids_to_tokens(ex['input_ids'])
	doc_words = []
	word_indices = []
	# Save whole words and the start and end indices of each word to their own lists
	for i, token in enumerate(doc_tokens):
		if token.startswith("▁"): # XLM-RoBERTa uses '▁' to denote word boundaries
			doc_words.append(token[1:])  # Remove the '▁' prefix
			word_indices.append((i, i + 1))
		elif doc_words and token not in set(punctuation) | SPECIAL_TOKENS:
			doc_words[-1] += token  # Append to the last word
			word_indices[-1] = (word_indices[-1][0], i + 1)
	doc_words = [re.sub(r"[^-\w\s'’]", "", w.lower()) for w in doc_words]
	keywords = [kw['token'] for kw in ex.get('keywords', [])]  # Preserves ordering of words
	for i, word in enumerate(doc_words):
		for j, kw in enumerate(keywords):
			if word == kw:
				ex['keywords'][j]['indices'] = word_indices[i]  # Save the indices of the keyword in the tokenized text
	# Add the tokenized text, words and word indices to the example
	return {
		"tokenized_text": doc_tokens,
		"words": doc_words,
		"word_indices": word_indices
	}

def flatten_by_tokens(state):
	"""Flatten a model layer output wrt. tokens. Tokens are in dimension -2."""
	return torch.mean(state, axis=-2).cpu().numpy().tolist()

def extract(model, tokenized, layer):
	"""Get hidden states of model for given layers."""
	tokenized.to(model.device)
	with torch.no_grad():
		output = model(**tokenized, output_hidden_states=True)
	embeddings = output['hidden_states'][layer].cpu().tolist()
	del output
	#return_value = [flatten_by_tokens(hidden_states[i]) for i in layers]
	torch.cuda.empty_cache()
	return embeddings

device = "cuda:0" if torch.cuda.is_available() else "cpu"

dataset = datasets.load_from_disk(hf_path)
tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")

#model = AutoModelForSequenceClassification.from_pretrained(
#	model_path,
#	output_hidden_states=True
#)
#model.to(device)

test = dataset['train'][0]
#tokenized = tokenize_fn(test)

# Map the dataset through tokenizer

dataset = dataset.map(tokenize_fn, batched=True)
dataset = dataset.map(enrich_dataset_fn, batched=False)

# TODO: test prediction and embedding extraction

#extracted = extract(model, tokenized, -1)

#print(extracted)

# TODO: fuzzy matching due to target token cleaning process (idea: tokenize targets and then match)
# TODO: save embeddings

#########################

# katenoi tokenisoidusta tekstistä sanat kokonaisiksi (tyyliin vaan if(t.startswith("_"))
# normalisoi samalla tavalla kuin kws.py
# exact matchaa keywordeihin

def get_index_list(example, kw):
	"""Get start and end indices of keywords in tokenized text."""
	return_indices = []
	for i, word in enumerate(example['words']):
		if word == kw:
			return_indices.append(example['word_indices'][i])
	return return_indices

print(dataset['train'][0])

sys.exit()

for kw in dataset['train'][0]['keywords']:
	print(get_index_list(dataset['train'][0], kw['token']))


encoded = tokenizer(
	test['text'],
	return_offsets_mapping=True,
	return_tensors="pt",
	truncation=True,
	padding=True
)
input_ids = encoded["input_ids"]
offsets = encoded["offset_mapping"][0]  # Shape: [num_tokens, 2]

#def normalize(text):
#	return re.sub(r"[^\w\s'’-]", "", text.lower())  # adjust to match your keyword pipeline

testkw = test['keywords'][0]['token'] # tässä voi testaa eri tokeneilla
keyword_tokens = tokenizer.tokenize(testkw)

#print(input_ids)
#print(offsets)

doc_tokens = tokenizer.convert_ids_to_tokens(encoded["input_ids"][0])
#print(doc_tokens)
#print(keyword_tokens)

words = []
word_indices = []
# Save whole words and the start and end indices of each word to their own lists
for i, token in enumerate(doc_tokens):
	if token.startswith("▁"): # XLM-RoBERTa uses '▁' to denote word boundaries
		words.append(token[1:])  # Remove the '▁' prefix
		word_indices.append((i, i + 1))
	elif words and token not in punctuation:
		words[-1] += token  # Append to the last word
		word_indices[-1] = (word_indices[-1][0], i + 1)

print(test['text'])
print(doc_tokens)
print(words)
print(word_indices)

#SACX regex clean
#tähän myös lowercase
words = list(map(lambda x: re.sub(r"[^-\w\s'’]", "", x.lower()), words))
print(words)
for i, word in enumerate(words):
	if word == testkw:
		print(word_indices[i])

print(test['keywords'])


SPECIAL_TOKENS = ["<s>", "</s>"]

print(doc_tokens[1])

for i, token in enumerate(doc_tokens):
	if token in SPECIAL_TOKENS:
		continue
	if token.startswith("▁"):
		print(f"Keyword found at index {i}")
		break
	else:
		print(f"Token: {token} at index {i}")


from collections import defaultdict

matched_token_indices = defaultdict(list)

def find_sublist(sub, full):
	"""Find all start indices where sub appears in full"""
	matches = []
	for i in range(len(full) - len(sub) + 1):
		# Use Levenshtein distance of 3 as a matching criterion to check for fuzzy matches
		#if edit_distance(full[i:i+len(sub)], sub) <= 3:
		if full[i:i+len(sub)] == sub:
			matches.append(list(range(i, i + len(sub))))
	return matches

indices = find_sublist(keyword_tokens, doc_tokens)
if indices:
	matched_token_indices[testkw].extend(indices)

for keyword in keywords:
	keyword_tokens = tokenizer.tokenize(keyword)
	indices = find_sublist(keyword_tokens, doc_tokens)
	if indices:
		matched_token_indices[keyword].extend(indices[0])  # or all if multiple matches


