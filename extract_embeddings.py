import sys
import os
import torch
import datasets
import pandas as pd
import re
import gc
from collections import defaultdict
from string import punctuation
from transformers import AutoTokenizer, AutoModelForSequenceClassification

SPECIAL_TOKENS = {"<s>", "</s>", "<pad>", "<unk>"}
SAFE_PUNCT = {"'", "’", "-"}  # punctuation we want to keep

# To make accessing path arguments easier
arg_keys = ['data_dir', 'model_dir', 'result_dir', 'lang', 'fold']
args = dict(zip(arg_keys, sys.argv[1:]))

hf_path = os.path.join(args['data_dir'], f"data-with-keywords-{args['lang']}")
model_path = os.path.join(args['model_dir'], f"{args['fold']}")
checkpoint_path = os.path.join(args['result_dir'], f"data-with-embs-{args['lang']}")
emb_path = os.path.join(args['result_dir'], args['lang'], f"embeddings-{args['fold']}.tsv")
os.makedirs(args['result_dir'], exist_ok=True)
os.makedirs(os.path.join(args['result_dir'], args['lang']), exist_ok=True)

def tokenize_fn(ex):
	if ex is None:
		return tokenizer("", return_tensors='pt', truncation=True, padding=True)
	return tokenizer(ex['text'], return_tensors='pt', truncation=True, padding=True)

def enrich_dataset_fn(ex):
	doc_tokens = tokenizer.convert_ids_to_tokens(ex['input_ids'])
	doc_words = []
	word_indices = []
	# Save whole words and the start and end indices of each word to their own lists
#	if ex['id'] in ["c07f9a7d90a5683303a3781b724ca05d", "112c38c5d2a87cabd9b96d409d688447", "3a25065f2a94edfa116a629516db8e77"]:
	print(doc_tokens)
	for i, token in enumerate(doc_tokens):
		if token.startswith("▁"): # XLM-RoBERTa uses '▁' to denote word boundaries
			doc_words.append(token[1:])  # Remove the '▁' prefix
			word_indices.append((i, i + 1))
		elif doc_words and (token in SAFE_PUNCT or token not in set(punctuation) | SPECIAL_TOKENS):
			doc_words[-1] += token  # Append to the last word
			word_indices[-1] = (word_indices[-1][0], i + 1)
	doc_words = [re.sub(r"[^-\w\s'’]", "", w.lower()) for w in doc_words]
	keywords = [kw['token'] for kw in ex.get('keywords', [])]
	for kw in ex.get("keywords", []):
		kw["indices_start"] = []
		kw["indices_end"] = []
	for i, word in enumerate(doc_words):
		for j, kw in enumerate(keywords):
			if word == kw:
				start, end = word_indices[i]
				ex["keywords"][j]["indices_start"].append(start)
				ex["keywords"][j]["indices_end"].append(end)
	for kw in ex["keywords"]:
		if not kw.get("indices_start"):
			print("No indices for:", kw["token"], "in example", ex["id"])
	return ex

def flatten_by_tokens(state):
	"""Flatten a model layer output wrt. tokens. Tokens are in dimension -2."""
	return torch.mean(state, axis=-2).cpu().numpy().tolist()

def extract(ex, layer=-1):
	"""Get hidden states of model for given layers."""
	input_ids = ex['input_ids'].to(model.device)
	attention_mask = ex['attention_mask'].to(model.device)
	# Add batch dimension if needed
	if input_ids.dim() == 1:
		input_ids = input_ids.unsqueeze(0)
	if attention_mask.dim() == 1:
		attention_mask = attention_mask.unsqueeze(0)
	with torch.no_grad():
		output = model(
			input_ids=input_ids,
			attention_mask=attention_mask,
			output_hidden_states=True
		)
	embeddings = output['hidden_states'][layer][0]
	del output
	torch.cuda.empty_cache()

	for kw in ex.get("keywords", []):
		starts = kw.get("indices_start", [])
		ends = kw.get("indices_end", [])
		span_embeddings = []

		for start, end in zip(starts, ends):
			token_embeds = embeddings[start:end]  # shape: [num_tokens_in_span, dim]
			span_embeddings.append(token_embeds)

		if not span_embeddings:
			continue  # no spans, skip
		
		all_embeds = torch.cat(span_embeddings, dim=0)  # shape: [total_tokens, dim]
		mean_embed = all_embeds.mean(dim=0).cpu().tolist()
		kw["mean_embed"] = mean_embed  # store in the original keyword dict
	
	for kw in ex["keywords"]:
		if "mean_embed" not in kw:
			print("No embedding for:", kw["token"], "in example", ex["id"])

	del embeddings
	gc.collect()

	return {"keywords": ex['keywords']}

def aggregate_and_save(dataset, lang, out_file):
	"""
	Aggregate embeddings across the whole dataset:
	one row per (keyword, pred) with global mean embedding.
	"""
	agg = defaultdict(list)

	# Step 1: collect embeddings
	for ex in dataset:
			for kw in ex.get("keywords", []):
					token = kw["token"]
					embed = torch.tensor(kw["mean_embed"])
					for pred in kw["preds"]:
							agg[(token, pred)].append(embed)

	# Step 2: average embeddings for each (keyword, pred)
	rows = []
	for (token, pred), embed_list in agg.items():
			all_embeds = torch.stack(embed_list)  # [n_occurrences, dim]
			mean_embed = all_embeds.mean(dim=0).tolist()
			rows.append({
					"lang": lang,
					"text": token,
					"embed_last": mean_embed,
					"preds": pred
			})

	# Step 3: build DataFrame and save
	df = pd.DataFrame(rows, columns=["lang", "text", "embed_last", "preds"])
	df.to_csv(out_file, sep="\t", index=False)
	print(f"✅ Aggregated {len(agg)} keyword-pred pairs → wrote {len(df)} rows to {out_file}")
	return df

device = "cuda:0" if torch.cuda.is_available() else "cpu"

dataset = datasets.load_from_disk(hf_path)
# Sample only 10 examples from the 'train' split for testing, retaining structure
dataset['train'] = dataset['train'].select(range(min(10, len(dataset['train']))))
tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")

model = AutoModelForSequenceClassification.from_pretrained(
	model_path,
	output_hidden_states=True
)
model.to(device)

dataset = dataset.map(tokenize_fn, batched=True)

# Now run the map with the updated features
dataset = dataset.map(enrich_dataset_fn, batched=False, remove_columns=[], load_from_cache_file=False)

dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])

dataset = dataset.map(extract, batched=False)

dataset.set_format("python")

print(dataset['train'][0])

dataset.save_to_disk(checkpoint_path)

aggregate_and_save(dataset['train'], args['lang'], emb_path)

