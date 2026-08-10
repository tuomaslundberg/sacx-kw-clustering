from sentence_transformers import SentenceTransformer
import sys
import os
import csv
import pandas as pd
from sklearn.preprocessing import normalize
import numpy as np
import ast

# To make accessing path arguments easier
arg_keys = ['keywords_path', 'model_path', 'output_dir', 'lang']
args = dict(zip(arg_keys, sys.argv[1:]))

lang = args['lang']
#model_file = f"cc.{lang}.300.bin"

keywords_path = os.path.join(args['keywords_path'])
model_path = args['model_path']
output_path = os.path.join(args['output_dir'], f"embeddings-{lang}.tsv")
os.makedirs(args['output_dir'], exist_ok=True)

#if not os.path.exists(model_path):
#	print(f"Model for language {lang} not found, downloading to script folder")
#	fasttext.util.download_model(lang, if_exists='ignore')
#	model_path = model_file

model = SentenceTransformer(model_path)

df_list = []

for file in os.listdir(keywords_path):
	if "unstable" not in file and file.endswith(".csv"):
		filepath = os.path.join(keywords_path, file)
		df_list.append(pd.read_csv(filepath))

# Combine dataframes and collect unique (token, pred) pairs
df = pd.concat(df_list, ignore_index=True)[['token', 'pred']]

def save_embeddings(lang, df, model, output_path):
	rows_df = df.copy()

	# ensure token strings
	rows_df['token'] = rows_df['token'].astype(str)
	rows_df['lang'] = lang

	# parse preds safely
	def parse_pred(p):
		try:
			if isinstance(p, str):
				val = ast.literal_eval(p)
			else:
				val = p
			return val[0] if (isinstance(val, (list, tuple)) and len(val) > 0) else None
		except Exception:
			return None

	rows_df['preds'] = rows_df['pred'].apply(parse_pred)

	# compute embeddings (numpy array)
	if rows_df.empty:
		# write header only
		with open(output_path, "w", encoding="utf-8", newline="") as f:
			writer = csv.writer(f, delimiter="\t")
			writer.writerow(["lang", "text", "embed_last", "preds"])
		return

	emb_list = [model.encode(t) for t in rows_df['token']]
	emb_arr = np.vstack(emb_list)  # shape (n, dim)

	# subtract mean embedding
	#mean_vec = emb_arr.mean(axis=0)
	#emb_arr = emb_arr - mean_vec

	emb_arr = normalize(emb_arr)

	# attach embeddings and text
	rows_df['text'] = rows_df['token']
	rows_df['embed_last'] = [arr.tolist() for arr in emb_arr]

	# prepare embed column as JSON strings and write TSV via pandas
	#rows_df['embed_last'] = rows_df['embed_last'].apply(lambda x: json.dumps(x, ensure_ascii=False))

	rows_df.loc[:, ['lang', 'text', 'embed_last', 'preds']].to_csv(
		output_path, sep='\t', index=False, encoding='utf-8'
	)

save_embeddings(lang, df, model, output_path)
print(f"Embeddings saved to {output_path}")
