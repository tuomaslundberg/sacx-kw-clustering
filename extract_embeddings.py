import sys
import os
import torch
import datasets
import regex as re
import unicodedata
import csv
import json
from collections import defaultdict
from transformers import AutoTokenizer, AutoModelForSequenceClassification

datasets.disable_progress_bar()

# keep the same normalization rule used for keywords
_KEEP_RE = re.compile(r"[-\w\s'’/<>]", re.UNICODE)
_KEEP_RE_ZH = re.compile(r"[\p{L}\p{N}'’\-/<>]", re.UNICODE)

# Characters to drop entirely (tatweel, ZWJ/ZWNJ, bidi marks, BOM)
_AR_UR_DROP = set(["\u0640", "\u200c", "\u200d", "\u200e", "\u200f", "\ufeff"])

# Map Arabic variants to Urdu canon (choose one target; here we pick: ک, ہ, ی)
_AR_UR_MAP = str.maketrans({
	"\u0643": "\u06A9",  # ك -> ک (kaf -> keheh)
	"\u064A": "\u06CC",  # ي -> ی (arabic yeh -> farsi yeh)
	"\u0649": "\u06CC",  # ى -> ی (alef maksura -> farsi yeh)
	"\u06D2": "\u06CC",  # ے -> ی (yeh barree -> farsi yeh)  <-- unify finals too
	"\u06D3": "\u06CC",  # ۓ -> ی
	"\u0647": "\u06C1",  # ه -> ہ (heh -> heh goal)
	"\u06BE": "\u06C1",  # ھ -> ہ (do chashmi heh -> heh goal)
	"\u06C2": "\u06C1",  # ۂ -> ہ (heh goal with hamza -> heh goal)
	"\u0629": "\u06C1",  # ة -> ہ (teh marbuta -> heh goal)
})

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

def normalize_arabic_urdu(s: str) -> str:
	# 1) Compatibility normalize
	s = unicodedata.normalize("NFKC", s)
	out = []
	for ch in s:
		if ch in _AR_UR_DROP:
			continue
		if unicodedata.category(ch) == "Mn":  # strip combining marks/harakat
			continue
		out.append(ch.translate(_AR_UR_MAP))
	s = "".join(out)
	# 2) collapse any Unicode whitespace to a single ASCII space
	s = re.sub(r"\p{Z}+", " ", s)
	# 3) remove everything except letters/numbers/hyphen/apostrophe/space/slash/angle-brackets
	s = re.sub(r"[^\p{L}\p{N}\-'’\s/<>]", "", s)
	return s

def _normalize_with_map(s: str):
	"""
	Return (norm_text, norm_to_orig_idx).
	norm_to_orig_idx[i] = original char index in `s` that produced norm_text[i].

	Uses per-character NFKC so expansions like 'ﬃ' -> 'ffi' are tracked.
	"""

	norm_chars = []
	idx_map = []

	if args['lang'] in {'ur', 'ar', 'fa'}:
		# Use NFC, not NFKC (avoid compatibility expansions)
		s_nfc = unicodedata.normalize("NFKC", s)
		for i, ch in enumerate(s_nfc):
			if ch in _AR_UR_DROP or unicodedata.category(ch) == "Mn":
				continue
			ch = ch.translate(_AR_UR_MAP)
			norm_chars.append(ch.lower())
			idx_map.append(i)

		norm = "".join(norm_chars)

		# whitespace collapse
		norm = re.sub(r"\p{Z}+", " ", norm)
		# strip everything except letters/numbers/hyphen/apostrophe/space/slash/angle-brackets
		norm = re.sub(r"[^\p{L}\p{N}\-'’\s/<>]", "", norm)

		return norm, idx_map[:len(norm)]  # ensure alignment

	else:
		# non-Arabic: just lowercase + filter
		norm_regex = _KEEP_RE_ZH if args['lang'] == 'zh' else _KEEP_RE
		for i, ch in enumerate(s):
			ch2 = ch.lower()
			if norm_regex.fullmatch(ch2):
				norm_chars.append(ch2)
				idx_map.append(i)
		return "".join(norm_chars), idx_map

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

def _fallback_token_search(k_norm, input_ids, tokenizer):
	"""
	Try to find keyword `k_norm` as contiguous tokens if offsets mapping fails.
	Returns (start, end) token indices or None.
	"""
	tokens = tokenizer.convert_ids_to_tokens(input_ids)
	norm_tokens = [re.sub(r"[^\p{L}\p{N}\-'’\s/<>]", "", t.lower()) for t in tokens]

	n = len(norm_tokens)
	k_parts = tokenizer.tokenize(k_norm)  # break keyword into subtokens

	for i in range(n - len(k_parts) + 1):
		if norm_tokens[i:i+len(k_parts)] == k_parts:
			return (i, i+len(k_parts))

	return None

def enrich_dataset_fn(batch):
	"""
	Batched version of enrichment:
	1) tokenize with offsets on ORIGINAL text,
	2) build a NORMALIZED copy of the text (same regex/lower as keywords) + index map,
	3) search normalized keywords in normalized text,
	4) map normalized char spans -> original char spans -> token spans.
	"""
	all_keywords = []  # output for this batch

	for text, ex_id, input_ids, offsets, kws in zip(
		batch["text"], batch["id"], batch["input_ids"], batch["offset_mapping"], batch["keywords"]
	):

		# find cutoff from offsets
		valid_offsets = [off for off in offsets if off[1] != 0]
		cutoff_char = 0 if not valid_offsets else valid_offsets[-1][1]

		truncated_text = text[:cutoff_char]

		# now normalize only the truncated text
		norm_text, norm_to_orig = _normalize_with_map(truncated_text)

		# process this example's keywords
		enriched_kws = []
		for kw in kws:  # kw is a dict
			if args['lang'] in {'ur','ar','fa'}:
				k_norm = normalize_arabic_urdu(kw["token"])
			elif args['lang'] == 'zh':
				k_norm = re.sub(r"[^\p{L}\p{N}'’\-/<>]", "", kw["token"].lower())
			else:
				k_norm = re.sub(r"[^-\w\s'’/<>]", "", kw["token"].lower())
			starts, ends = [], []
			pos = 0
			while True:
				pos = norm_text.find(k_norm, pos)
				if pos == -1:
					break
				norm_start = pos
				norm_end   = pos + len(k_norm)

				# 4a) normalized span -> original char span
				char_start = norm_to_orig[norm_start]
				char_end   = norm_to_orig[norm_end - 1] + 1

				# 4b) original char span -> token span
				span = _char_span_to_token_span(
					char_start, char_end, input_ids, offsets, SPECIAL_IDS
					)
				if span is None:
					span = _fallback_token_search(k_norm, input_ids, tokenizer)
				if span is not None:
					s, e = span
					starts.append(s)
					ends.append(e)

				pos += 1  # allow overlapping matches

			kw = dict(kw)  # make a copy so we don't mutate in place
			kw["indices_start"] = starts
			kw["indices_end"]   = ends
			enriched_kws.append(kw)

		all_keywords.append(enriched_kws)

	return {"keywords": all_keywords}

def extract(batch, layer=-1):
	"""
	Batched version: compute hidden states for a batch of examples.
	"""

	# Convert to torch tensors (2D already: [B, L])
	input_ids = torch.as_tensor(batch["input_ids"], device=model.device)
	attention_mask = torch.as_tensor(batch["attention_mask"], device=model.device)

	with torch.no_grad():
		output = model(
			input_ids=input_ids,
			attention_mask=attention_mask,
			output_hidden_states=True,
		)
		embeddings = output["hidden_states"][layer]  # [B, L, D]
	del output
	torch.cuda.empty_cache()

	new_keywords = []
	batch_size = input_ids.size(0)

	for i in range(batch_size):
		example_keywords = []
		for kw in batch["keywords"][i]:
			starts = kw.get("indices_start", [])
			ends = kw.get("indices_end", [])
			span_embeddings = []

			for start, end in zip(starts, ends):
				token_embeds = embeddings[i, start:end]  # [span_len, dim]
				if token_embeds.numel() > 0:
					span_embeddings.append(token_embeds)

			if span_embeddings:
				all_embeds = torch.cat(span_embeddings, dim=0)  # [total_tokens, dim]
				mean_embed = all_embeds.mean(dim=0).cpu().tolist()
				kw["mean_embed"] = mean_embed
			else:
				print("No embedding for:", kw["token"], "in batch example", batch["id"][i], flush=True)

			example_keywords.append(kw)

		new_keywords.append(example_keywords)

	return {"keywords": new_keywords}

def aggregate_and_save(dataset, lang, output_path):
	agg = defaultdict(list)
	stats = {
		"examples": 0,
		"keywords_total": 0,
		"keywords_with_embed": 0,
		"keywords_without_embed": 0,
		"embeds_nan_or_inf": 0,
	}

	def to_float_tensor(x):
		try:
			t = torch.tensor(x, dtype=torch.float32)
			if torch.isnan(t).any() or torch.isinf(t).any():
				return None
			return t
		except Exception:
			return None

	for ex in dataset:
		stats["examples"] += 1
		kws = ex.get("keywords", []) or []
		for kw in kws:
			stats["keywords_total"] += 1
			token = kw.get("token")

			me = kw.get("mean_embed", None)
			if me is None:
				stats["keywords_without_embed"] += 1
				continue
			if not isinstance(me, (list, tuple)) or len(me) == 0:
				stats["keywords_without_embed"] += 1
				continue

			embed = to_float_tensor(me)
			if embed is None:
				stats["embeds_nan_or_inf"] += 1
				stats["keywords_without_embed"] += 1
				continue

			preds = kw.get("preds", None)
			if preds is None:
				# If there are no preds, we can still store under a special label
				preds_iter = ["<NO_PRED>"]
			elif isinstance(preds, (list, tuple)):
				preds_iter = preds
			else:
				preds_iter = [preds]

			for pred in preds_iter:
				agg[(token, pred)].append(embed)
			stats["keywords_with_embed"] += 1

	# Aggregate
	rows = []
	for (token, pred), embed_list in agg.items():
		if not embed_list:
			continue
		all_embeds = torch.stack(embed_list)  # [n_occurrences, dim]
		mean_embed = all_embeds.mean(dim=0).tolist()
		rows.append({
			"lang": lang,
			"text": token,
			"embed_last": mean_embed,
			"preds": pred
		})

	# Write TSV
	with open(output_path, "w", encoding="utf-8", newline="") as f:
		writer = csv.writer(f, delimiter="\t")
		writer.writerow(["lang", "text", "embed_last", "preds"])
		for r in rows:
			writer.writerow([
				r["lang"],
				r["text"],
				json.dumps(r["embed_last"], ensure_ascii=False),
				r["preds"],
			])

	# Helpful summary to stdout
	print(
		f"[aggregate_and_save] examples={stats['examples']}, "
		f"kws_total={stats['keywords_total']}, "
		f"kws_with_embed={stats['keywords_with_embed']}, "
		f"kws_without_embed={stats['keywords_without_embed']}, "
		f"nan_or_inf={stats['embeds_nan_or_inf']}, "
		f"unique_pairs={len(agg)}, rows_written={len(rows)}, "
		f"out={output_path}",
		flush=True
	)

device = "cuda:0" if torch.cuda.is_available() else "cpu"

dataset = datasets.load_from_disk(hf_path)

# Uncomment for testing
#dataset['train'] = dataset['train'].select(range(min(10, len(dataset['train']))))

tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base", use_fast=True)

SPECIAL_IDS = set(tokenizer.all_special_ids)

model = AutoModelForSequenceClassification.from_pretrained(
	model_path,
	output_hidden_states=True
)
model.to(device)

print("starting tokenisation", flush=True)
dataset = dataset.map(tokenize_fn, batched=True)

print("tokenisation done, starting enrichment", flush=True)
# Now run the map with the updated features
dataset = dataset.map(enrich_dataset_fn, batched=True, num_proc=8)

print("enrichment done, starting extraction", flush=True)
dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])

dataset = dataset.map(extract, batched=True, batch_size=32)
print("extraction done", flush=True)
dataset.set_format("python")

aggregate_and_save(dataset['train'], args['lang'], emb_path)

