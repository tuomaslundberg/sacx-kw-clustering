import sys
import os
import torch
import datasets
import pandas as pd
#import re
import regex as re
import gc
import unicodedata
import bisect
from collections import defaultdict
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Disable datasets progress bar output
datasets.utils.logging.set_verbosity_error()

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

old_problem_ids = [
	'a9f4d23fe474fcb914037351e5c48c83',
	'e150d5cabcc548c42c9c20ac1e551e25', # the new one for refrain
	'a6e686a07fc27eae5949cb09b0d54b1e',
	'316abb7338d7a4f2c4b74f269c1fa6b0',
	'1f34c0d166297eeb69366f2827745d44', # urdu starts here
	'ace811d47f9830cd7d416479115e558f',
	'b0983919c97fea90c5f70190a76ae94f',
	'9bb3baeecef8d8fcd59d8b9d05ca5f58',
	'57359c9725687a73a8f3a67899ef06c5',
	'55ca597f56a0cbe25b5b9c824a3dd2ff',
	'297a8be08cd58ee1ba5f6aa0fb02a850',
	'deffd76b849fc30d59970e4f1e112a14',
	'96398054c49654150ae8330076093fb2',
	'b762c39f710d59056a13292dae2957f2',
	'7cc020b865c623b8f6a1b5deea35865e',
	'a318c2e11e9fec3f3b936396ef11f3d1',
	'ded507eeefa4f4035155da05f0bfc5a1',
	'b161f541c71116745e90b45933758e89',
	'c5217be354e2584c5b356b8e17ff7d72',
	'260c5cafee2a882e35962b37341660b1',
	'660582dcffb207df331f292b8d012862',
	'c5e38c72df9ea754cbbae829deff54a9',
	'775bed68f8ef364a15228f6f1310d5fa',
	'29ee7e35b897984cba9d4a9b5030f42a',
	'a391e126deed3c6855e566690ceb0802',
	'd106e8e1d706e231b22c1d365573206f',
	'45d434b1d7b1716e9b8549f4304c2e40',
	'13de132b1f42fba21d5b49ec70dbd36e',
	'0fdeae551abdc84d9ceabc4257d72dfe',
	'3f273f11e41d42c7d68eb68485047231',
	'a847cf81377e234cc4d7b8a355d0ce93',
	'0a58f01b2c4e5fa6f0ef7161ea2e5024',
	'ee62426972588120d7f040a299c0b152',
	'710dbbd0a732a5b90d87ae5e89f74e60',
	'608c5a28c581c28b48327dcb97ff788f',
	'af110f31caae3e0ae7649dba957659b6',
	'3b909e9144c0f928b288917e2358339f',
	'6efd3c961bef085e68b59eecd240c8f9',
	'aef9aff719f7381e9adc9b3a166b165e',
	'5c09e33b7caadf8435a70901e2307f6e',
	'0abfebe2703f48613ec84517238a5752',
	'60ef55df2aa61afb330f01950a6d06d6',
	'd6e46ad30f40c7b6a2a74b9892b74b65',
	'61ebd91402479d6cf3494529d89739c3',
	'0044b8332b6b42cee2e777771b89e6a0',
	'71e292571f836085a776b475f27bd50e',
	'ca047269e5630aa3c551f9357695b576',
	'c29dec126da00e269af154bc016ce785',
	'85d0cba8966f6fc6185f43708b8278ca',
	'6845713408d93c0a71cfb79883848d51',
	'c17367c82b3ab1ef3dba13e9524474be',
	'b7fc0a170f92f06328f70a234c1e282f',
	'974cc08d80bee55843f1ce8ce793705c',
	'71846ffe5bf514d15b444856359a8588',
	'f9463991b9787200bfb9277d16044eab',
	'e44e9f3626efe7f888bbb69bf2a5d707',
	'ea6665b0602e858dab141076e29b33a3',
	'e306822d139f08df83ac8bdcbf354529',
	'973f6d1f6445ec077c938c8d28e1341a',
	'2a752f194c42faa634f973857b838700',
	'c3a979ba30f0d3ca73bd0ed6657fa065',
	'ac7be3e425b0db1d9f8da9533ac40446',
	'ad9aab99f1bc6c2d807b1a7964a8d69a',
	'2ac492c3bbc70061bf8e74ea13921e02',
	'37e986500d272c8c76dbc44aafd2918c',
	'f6a2407a56fb0ca1e06c0a8286af72ec',
	'9d90c4d686eba3f98eb1fdc750beaf28',
	'27b348b0bd4cca7e80b885f94ea716d7',
	'2f8f1778cf2b7f36df49f319bfc58f35',
	'b81524fec54103b3a244492aa98c5fda',
	'30ec6b95083a1d724e19a659d4c378df',
	'bd55f8598147cf5680b31942b1e3630e',
	'fae8ee3c8cde0d6948d3bac7a97b4d63', # chinese starts here
	'9b28775dbccbba103ce9c5dfe1b3d5d0',
	'386e347355c76759c6389c6667bf8005',
	'7d6bf575a51f437339670ca605cb9050',
	'345faffd89ca98207c89658a901a320f',
	'd744c42c00378c7ab41d9a4e9b3bc275'
]

old2_problem_ids = [
	'd255da76c0532de49b8593bbe5bd016f',
	'5a822a1e0d87518f927d8f64dfcb8d79',
	'6778f1ae7ff274729f6054cfdbad351b',
	'9bb3baeecef8d8fcd59d8b9d05ca5f58',
	'57359c9725687a73a8f3a67899ef06c5',
	'55ca597f56a0cbe25b5b9c824a3dd2ff',
	'297a8be08cd58ee1ba5f6aa0fb02a850',
	'd5f8b19e53c4655a1a30dc46ab4b275f',
	'b762c39f710d59056a13292dae2957f2',
	'7cc020b865c623b8f6a1b5deea35865e',
	'a318c2e11e9fec3f3b936396ef11f3d1',
	'ded507eeefa4f4035155da05f0bfc5a1',
	'b161f541c71116745e90b45933758e89',
	'c5217be354e2584c5b356b8e17ff7d72',
	'e807a6f0763a6f1e33ee164cf954127b',
	'4583e8800a7298e70c2af69dc54755c9',
	'b022d46c2fdbfb5fb903104205d6791a',
	'660582dcffb207df331f292b8d012862',
	'c5e38c72df9ea754cbbae829deff54a9',
	'a992d2a2b0ab34afee0cb2e3b755efc0',
	'775bed68f8ef364a15228f6f1310d5fa',
	'29ee7e35b897984cba9d4a9b5030f42a',
	'da0427354e12af9d4c939a805799b803',
	'1a5064e8d1d6b32bbe8c19ac93c08e32',
	'a391e126deed3c6855e566690ceb0802',
	'45d434b1d7b1716e9b8549f4304c2e40',
	'a537d25c82c348de3e2a11e1682efa55',
	'2b191ac9fc8393f404aa2b49ba0802a6',
	'4c4ff810a3244ddf679060a2172eb86a',
	'a9a60821727893f9c58a8a90ecf89124',
	'2caaeee1522936180f1604f7d336fa19',
	'13de132b1f42fba21d5b49ec70dbd36e',
	'0fdeae551abdc84d9ceabc4257d72dfe',
	'3f273f11e41d42c7d68eb68485047231',
	'13a248669f9a4957de24e39cd857c08e',
	'7851bdad75b6b1df171f7b40281e3db7',
	'a847cf81377e234cc4d7b8a355d0ce93',
	'0a58f01b2c4e5fa6f0ef7161ea2e5024',
	'015ff2e53141a98c2bae00576ccd1e67',
	'710dbbd0a732a5b90d87ae5e89f74e60',
	'608c5a28c581c28b48327dcb97ff788f',
	'af110f31caae3e0ae7649dba957659b6',
	'3b909e9144c0f928b288917e2358339f',
	'6efd3c961bef085e68b59eecd240c8f9',
	'cd8b3bd3ce633bbc91b1782733ae6ebd',
	'aef9aff719f7381e9adc9b3a166b165e',
	'db5db473a3744a1f2dd1d53177fc638e',
	'5c09e33b7caadf8435a70901e2307f6e',
	'756b2b0443263350b139b381657d2e82',
	'c2ef3dc5a5003e390f2b00e97cf97657',
	'eba91dd4359d39b63301d6b5329877a0',
	'0abfebe2703f48613ec84517238a5752',
	'4cfee8c9b4eb5af91d3a45b3f5db587e',
	'c081da5cec2b06059a9f18f5d617104a',
	'727d8c894cec313f1273546e5e942f18',
	'd6e46ad30f40c7b6a2a74b9892b74b65',
	'61ebd91402479d6cf3494529d89739c3',
	'0044b8332b6b42cee2e777771b89e6a0',
	'6438088c20278db740e14f9225722d48',
	'8ecd296e935c9413a4a48db4584f9498',
	'ff453282a32e99766502abb0de94c730',
	'71e292571f836085a776b475f27bd50e',
	'ca047269e5630aa3c551f9357695b576',
	'496f4934dfdfa58a98efd676af126668',
	'c29dec126da00e269af154bc016ce785',
	'6845713408d93c0a71cfb79883848d51',
	'ce32dfe61c4f40d61bf305b302ef6083',
	'600509b565f58ade9597ba1c3c0fee87',
	'b7fc0a170f92f06328f70a234c1e282f',
	'7b553e00911e9cf9a0943416d949e478',
	'974cc08d80bee55843f1ce8ce793705c',
	'17174a5cd4591832d7d73f630f4b8938',
	'9fa1da527d93c2a93dc556150419cbf8',
	'71846ffe5bf514d15b444856359a8588',
	'f9463991b9787200bfb9277d16044eab',
	'e44e9f3626efe7f888bbb69bf2a5d707',
	'ea6665b0602e858dab141076e29b33a3',
	'f11e8017c39b96801ba48a3b2d030fdf',
	'973f6d1f6445ec077c938c8d28e1341a',
	'68da0f8f47006b79fa104ead924aaf46',
	'2a752f194c42faa634f973857b838700',
	'0ba594a485aaf5c01f0a61c508a33669',
	'c3a979ba30f0d3ca73bd0ed6657fa065',
	'9b634d2b691a45471625f037f9e23bdd',
	'18545b2fecedff3eca910012c55ce3b2',
	'e6420129e210e80d1fbd3df2d6da1a1a',
	'308f1b0d9e9be623a4d050f6abe84735',
	'ac7be3e425b0db1d9f8da9533ac40446',
	'ad9aab99f1bc6c2d807b1a7964a8d69a',
	'466e8795735ec9e8c2195df5fa026b20',
	'2ac492c3bbc70061bf8e74ea13921e02',
	'440cd07ca933bb3a92c04c271f6b6120',
	'37e986500d272c8c76dbc44aafd2918c',
	'b8216b521cf458e5135664ee3f03cac3',
	'9d90c4d686eba3f98eb1fdc750beaf28',
	'27b348b0bd4cca7e80b885f94ea716d7',
	'b81524fec54103b3a244492aa98c5fda',
	'b24efc24b867530c6a40560878eb5fcd',
	'aac6656f3fea67c2aba4dc35310c6b2d'
]

problem_ids = [
	'9bb3baeecef8d8fcd59d8b9d05ca5f58',
	'57359c9725687a73a8f3a67899ef06c5',
	'55ca597f56a0cbe25b5b9c824a3dd2ff',
	'297a8be08cd58ee1ba5f6aa0fb02a850',
	'b762c39f710d59056a13292dae2957f2',
	'7cc020b865c623b8f6a1b5deea35865e',
	'a318c2e11e9fec3f3b936396ef11f3d1',
	'ded507eeefa4f4035155da05f0bfc5a1',
	'b161f541c71116745e90b45933758e89',
	'c5217be354e2584c5b356b8e17ff7d72',
	'e807a6f0763a6f1e33ee164cf954127b',
	'4583e8800a7298e70c2af69dc54755c9',
	'b022d46c2fdbfb5fb903104205d6791a',
	'660582dcffb207df331f292b8d012862',
	'c5e38c72df9ea754cbbae829deff54a9',
	'775bed68f8ef364a15228f6f1310d5fa',
	'29ee7e35b897984cba9d4a9b5030f42a',
	'1a5064e8d1d6b32bbe8c19ac93c08e32',
	'a391e126deed3c6855e566690ceb0802',
	'45d434b1d7b1716e9b8549f4304c2e40',
	'a9a60821727893f9c58a8a90ecf89124',
	'13de132b1f42fba21d5b49ec70dbd36e',
	'0fdeae551abdc84d9ceabc4257d72dfe',
	'3f273f11e41d42c7d68eb68485047231',
	'a847cf81377e234cc4d7b8a355d0ce93',
	'0a58f01b2c4e5fa6f0ef7161ea2e5024',
	'710dbbd0a732a5b90d87ae5e89f74e60',
	'608c5a28c581c28b48327dcb97ff788f',
	'af110f31caae3e0ae7649dba957659b6',
	'3b909e9144c0f928b288917e2358339f',
	'6efd3c961bef085e68b59eecd240c8f9',
	'cd8b3bd3ce633bbc91b1782733ae6ebd',
	'aef9aff719f7381e9adc9b3a166b165e',
	'5c09e33b7caadf8435a70901e2307f6e',
	'eba91dd4359d39b63301d6b5329877a0',
	'0abfebe2703f48613ec84517238a5752',
	'd6e46ad30f40c7b6a2a74b9892b74b65',
	'61ebd91402479d6cf3494529d89739c3',
	'0044b8332b6b42cee2e777771b89e6a0',
	'71e292571f836085a776b475f27bd50e',
	'ca047269e5630aa3c551f9357695b576',
	'496f4934dfdfa58a98efd676af126668',
	'c29dec126da00e269af154bc016ce785',
	'6845713408d93c0a71cfb79883848d51',
	'ce32dfe61c4f40d61bf305b302ef6083',
	'b7fc0a170f92f06328f70a234c1e282f',
	'974cc08d80bee55843f1ce8ce793705c',
	'17174a5cd4591832d7d73f630f4b8938',
	'9fa1da527d93c2a93dc556150419cbf8',
	'71846ffe5bf514d15b444856359a8588',
	'f9463991b9787200bfb9277d16044eab',
	'e44e9f3626efe7f888bbb69bf2a5d707',
	'ea6665b0602e858dab141076e29b33a3',
	'973f6d1f6445ec077c938c8d28e1341a',
	'2a752f194c42faa634f973857b838700',
	'c3a979ba30f0d3ca73bd0ed6657fa065',
	'18545b2fecedff3eca910012c55ce3b2',
	'308f1b0d9e9be623a4d050f6abe84735',
	'ac7be3e425b0db1d9f8da9533ac40446',
	'ad9aab99f1bc6c2d807b1a7964a8d69a',
	'2ac492c3bbc70061bf8e74ea13921e02',
	'37e986500d272c8c76dbc44aafd2918c',
	'b8216b521cf458e5135664ee3f03cac3',
	'9d90c4d686eba3f98eb1fdc750beaf28',
	'27b348b0bd4cca7e80b885f94ea716d7',
	'b81524fec54103b3a244492aa98c5fda'
]

'''
print("="*80, flush=True)
print("\nIn both:\n", flush=True)
for id in problem_ids:
	if id in old_problem_ids:
		print(id, flush=True)
print("\n", flush=True)
print("="*80, flush=True)
'''
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
		#s_nfc = s
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
		#print(f"Token ID: {tid}, Offsets: ({a}, {b})")
		if tid in special_ids or b <= a:
			#print("b <= a")
			continue
		# token intersects the char span?
		if b <= char_start:
			#print("b <= char_start")
			continue
		if a >= char_end:
			#print("a >= char_end")
			if tok_start is not None:
				#print("tok_start is not None")
				break
			else:
				#print("tok_start is None")
				continue
		if tok_start is None:
			#print("tok_start is None")
			tok_start = ti
		tok_end = ti + 1
	if tok_start is None or tok_end is None or tok_start >= tok_end:
		return None
	#print(f"tok_start: {tok_start}, tok_end: {tok_end}")
	return (tok_start, tok_end)

def byte_offsets_to_char_offsets(offsets, text):
	"""
	Convert offsets given in UTF-8 byte indices to Python string char indices.
	offsets: list of (start_byte, end_byte)
	text: original Python str
	Returns list of (start_char, end_char)
	"""
	if not offsets:
		return offsets

	# cumulative bytes length at each char boundary:
	byte_prefix = [0]  # byte_prefix[i] = number of bytes in text[:i]
	enc = text.encode("utf-8")
	# build cumulative by iterating characters (cheap enough)
	for ch in text:
		byte_prefix.append(byte_prefix[-1] + len(ch.encode("utf-8")))

	# convert each (a,b) byte pair to char indices
	char_offsets = []
	for a, b in offsets:
		# Clip a,b so they are <= total bytes (defensive)
		a = max(0, min(a, len(enc)))
		b = max(0, min(b, len(enc)))
		# bisect_right(byte_prefix, x) - 1 gives char index whose prefix <= x
		# we want the index of the character that starts at or before byte index a
		start_char = bisect.bisect_right(byte_prefix, a) - 1
		end_char   = bisect.bisect_right(byte_prefix, b) - 1
		# defensive: ensure non-negative
		start_char = max(0, start_char)
		end_char = max(0, end_char)
		char_offsets.append((start_char, end_char))
	return char_offsets

'''
def enrich_dataset_fn(ex):
	"""
	Works for Chinese and non-Chinese:
	1) tokenize with offsets on ORIGINAL text,
	2) build a NORMALIZED copy of the text (same regex/lower as keywords) + index map,
	3) search normalized keywords in normalized text,
	4) map normalized char spans -> original char spans -> token spans.
	"""
	text = ex["text"]

	if ex['id'] in problem_ids:
		print(text)

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
'''

def debug_keyword_match(text, kw, tokenizer):
	print("="*80)
	print(f"DEBUGGING keyword: {kw}")
	print("- Original text snippet:")
	print(text[:300], "..." if len(text) > 300 else "")

	# Unicode names of the keyword
	print("\n- Keyword chars + Unicode names:")
	for ch in kw:
		print(f"  '{ch}' U+{ord(ch):04X} {unicodedata.name(ch, 'UNKNOWN')}")

	# Normalize text and keyword
	kw_nfc   = unicodedata.normalize("NFC", kw)
	text_nfc = unicodedata.normalize("NFC", text)

	print("\n- In raw text?:", kw in text)
	print("- In NFC text?:", kw_nfc in text_nfc)

	# Regex-cleaning step
	def clean(s):
		cleaned = re.sub(r"[^\p{L}\p{N}\-'\s/<>]", "", s.lower()) if args['lang'] in ['zh', 'ur'] else re.sub(r"[^-\w\s'’/<>]", "", s.lower())
		#return re.sub(r"[^\p{L}\p{N}\-'\s]", "", s.lower())
		#return re.sub(r"[^-\w\s'’/<>]", "", s.lower())
		return cleaned

	text_clean = clean(text_nfc)
	kw_clean   = clean(kw_nfc)

	print("\n- Cleaned keyword:", kw_clean)
	print("- In cleaned text?:", kw_clean in text_clean)

	# Tokenization
	tokens = tokenizer.tokenize(text)
	ids	= tokenizer(text, return_offsets_mapping=True)
	offsets = ids["offset_mapping"]

	print("\n- First 512 tokens:", tokens[:512])
	print("- Offsets:", offsets)

	# Try finding keyword characters in tokens
	found_in_tokens = [i for i,tok in enumerate(tokens) if kw_clean in tok]
	print("\n- Keyword appears in token(s):", found_in_tokens)

	print("="*80)

def debug_from_token_idx(text, offsets, idx, width=20):
	a, b = offsets[idx]
	a = max(0, a - width); b = min(len(text), b + width)
	frag = text[a:b]
	print(f"\nToken @{idx} context [{a}:{b}]: {frag!r}")
	for ch in frag:
		print(f"  '{ch}' U+{ord(ch):04X} {unicodedata.name(ch, 'UNKNOWN')}")

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
		#if ex_id in problem_ids:
		#	print(text, flush=True)
		#	[debug_keyword_match(text, kw['token'], tokenizer) for kw in kws]# if kw['token'] in ['video', 'refrain']]

		#print(text[:200])
		#for i in range(10):
		#	print("slice by chars:", text[offsets[i][0]:offsets[i][1]])
		#	print("slice by bytes:", text.encode('utf-8')[offsets[i][0]:offsets[i][1]])
		#return
		
		#offsets = byte_offsets_to_char_offsets(offsets, text)

		# find cutoff from offsets
		valid_offsets = [off for off in offsets if off[1] != 0]
		cutoff_char = 0 if not valid_offsets else valid_offsets[-1][1]

		#print(text)
		truncated_text = text[:cutoff_char]
		#print(cutoff_char)
		#print(truncated_text)

		# now normalize only the truncated text
		norm_text, norm_to_orig = _normalize_with_map(truncated_text)
		#if ex_id in problem_ids:
		#print("Text ID:", ex_id)
		#print("Normalised text:", norm_text)

		# process this example's keywords
		enriched_kws = []
		for kw in kws:  # kw is a dict
			if args['lang'] in {'ur','ar','fa'}:
				k_norm = normalize_arabic_urdu(kw["token"])
			elif args['lang'] == 'zh':
				k_norm = re.sub(r"[^\p{L}\p{N}'’\-/<>]", "", kw["token"].lower())
			else:
				k_norm = re.sub(r"[^-\w\s'’/<>]", "", kw["token"].lower())
			#if ex_id in problem_ids:
			#	print("Normalised keyword:", k_norm)
			starts, ends = [], []
			pos = 0
			while True:
				pos = norm_text.find(k_norm, pos)
				#print("k_norm:", k_norm)
				#print("pos:", pos)
				#if ex_id == 'd255da76c0532de49b8593bbe5bd016f':
				#	print(pos)
				if pos == -1:
					break
				norm_start = pos
				norm_end   = pos + len(k_norm)

				# 4a) normalized span -> original char span
				char_start = norm_to_orig[norm_start]
				char_end   = norm_to_orig[norm_end - 1] + 1

				#if ex_id == 'd255da76c0532de49b8593bbe5bd016f':
				#print("Char start:", char_start)
				#print("Char end:", char_end)
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

		#if ex_id in problem_ids:
			#print(enriched_kws)
		all_keywords.append(enriched_kws)

	return {"keywords": all_keywords}

def extract(batch, layer=-1):
	"""
	Batched version: compute hidden states for a batch of examples.
	"""

	# Convert to torch tensors (2D already: [B, L])
	input_ids = torch.tensor(batch["input_ids"]).to(model.device)
	attention_mask = torch.tensor(batch["attention_mask"]).to(model.device)

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
				print("No embedding for:", kw["token"], "in batch example", batch["id"][i])

			example_keywords.append(kw)

		new_keywords.append(example_keywords)

	return {"keywords": new_keywords}

'''
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
			mean_embed = all_embeds.mean(dim=0).tolist() if all_embeds is not None else None
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
'''

import csv
import json
import torch
from collections import defaultdict

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
		f"out={output_path}"
	)

device = "cuda:0" if torch.cuda.is_available() else "cpu"

dataset = datasets.load_from_disk(hf_path)

# Uncomment for testing
# dataset['train'] = dataset['train'].select(range(min(10, len(dataset['train']))))
#dataset['train'] = dataset['train'].filter(lambda x: x['id'] in problem_ids)
#dataset['train'] = dataset['train'].filter(lambda x: x['id'] == '9bb3baeecef8d8fcd59d8b9d05ca5f58')

tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base", use_fast=True)

SPECIAL_IDS = set(tokenizer.all_special_ids)

model = AutoModelForSequenceClassification.from_pretrained(
	model_path,
	output_hidden_states=True
)
model.to(device)
print("starting tokenisation")
dataset = dataset.map(tokenize_fn, batched=True)
print("tokenisation done, starting enrichment")
# Now run the map with the updated features
dataset = dataset.map(enrich_dataset_fn, batched=True, num_proc=8)#False, remove_columns=[], load_from_cache_file=False)
#sys.exit(0)
print("enrichment done, starting extraction")
dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])

dataset = dataset.map(extract, batched=True, batch_size=32)
print("extraction done")
dataset.set_format("python")

#dataset.save_to_disk(checkpoint_path)

aggregate_and_save(dataset['train'], args['lang'], emb_path)

