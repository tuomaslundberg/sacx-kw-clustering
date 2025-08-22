import sys
import os
import torch
import datasets
import pandas as pd
import re
import gc
from collections import defaultdict
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# keep the same normalization rule used for keywords
_KEEP_RE = re.compile(r"[-\w\s'’]", re.UNICODE)

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
	text = "" if ex is None else ex['text']
	return tokenizer(
		text,
		return_tensors='pt',
		return_offsets_mapping=True,
		truncation=True,
		padding=True,
		add_special_tokens=True
	)

def _normalize_with_map(s: str):
	"""
	Return (norm_text, norm_to_orig_idx).
	norm_to_orig_idx[i] = original char index in s of norm_text[i].
	"""
	out = []
	idx_map = []
	for i, ch in enumerate(s):
		ch2 = ch.lower()
		if _KEEP_RE.fullmatch(ch2):
			out.append(ch2)
			idx_map.append(i)
	return "".join(out), idx_map

def _char_span_to_token_span(char_start, char_end, input_ids, offsets, special_ids):
	"""
	Map a [char_start, char_end) span in the ORIGINAL text to a token span [t0, t1).
	Skips special tokens and empty-offset tokens.
	"""
	tok_start = tok_end = None
	for ti, (tid, (a, b)) in enumerate(zip(input_ids, offsets)):
		if tid in special_ids or b <= a:
			continue
		# token intersects the char span?
		if b <= char_start:
			continue
		if a >= char_end:
			if tok_start is not None:
				break
			else:
				continue
		if tok_start is None:
			tok_start = ti
		tok_end = ti + 1
	if tok_start is None or tok_end is None or tok_start >= tok_end:
		return None
	return (tok_start, tok_end)

def enrich_dataset_fn(ex):
	"""
	Works for Chinese and non-Chinese:
	1) tokenize with offsets on ORIGINAL text,
	2) build a NORMALIZED copy of the text (same regex/lower as keywords) + index map,
	3) search normalized keywords in normalized text,
	4) map normalized char spans -> original char spans -> token spans.
	"""
	text = ex["text"]

	input_ids = ex["input_ids"]
	offsets   = ex["offset_mapping"]

	# 2) normalized text + char index map
	norm_text, norm_to_orig = _normalize_with_map(text)

	# 3) for each keyword, normalize the keyword and search in normalized text
	for kw in ex.get("keywords", []):
		k_norm = kw["token"]
		starts, ends = [], []
		pos = 0
		while True:
			pos = norm_text.find(k_norm, pos)
			if pos == -1:
				break
			norm_start = pos
			norm_end   = pos + len(k_norm)

			# 4a) map normalized span back to ORIGINAL char span
			char_start = norm_to_orig[norm_start]
			char_end   = norm_to_orig[norm_end - 1] + 1

			# 4b) original char span -> token span
			span = _char_span_to_token_span(
				char_start, char_end, input_ids, offsets, SPECIAL_IDS
			)
			if span is not None:
				s, e = span
				starts.append(s)
				ends.append(e)

			pos += 1  # allow overlapping matches

		kw["indices_start"] = starts
		kw["indices_end"]   = ends

	return {"keywords": ex["keywords"]}

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

# Uncomment for testing
# dataset['train'] = dataset['train'].select(range(min(10, len(dataset['train']))))

tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base", use_fast=True)

SPECIAL_IDS = set(tokenizer.all_special_ids)

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

dataset.save_to_disk(checkpoint_path)

aggregate_and_save(dataset['train'], args['lang'], emb_path)

