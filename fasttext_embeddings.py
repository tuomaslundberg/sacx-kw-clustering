import sys
import os
import csv
import json
import pandas as pd
import fasttext
import fasttext.util

# To make accessing path arguments easier
arg_keys = ['keywords_path', 'model_path', 'output_dir', 'lang']
args = dict(zip(arg_keys, sys.argv[1:]))

lang = args['lang']
model_file = f"cc.{lang}.300.bin"

keywords_path = os.path.join(args['keywords_path'])
model_path = os.path.join(args['model_path'], model_file)
output_path = os.path.join(args['output_dir'], f"embeddings-{lang}.tsv")
os.makedirs(args['output_dir'], exist_ok=True)

if not os.path.exists(model_path):
	print(f"Model for language {lang} not found, downloading to script folder")
	fasttext.util.download_model(lang, if_exists='ignore')
	model_path = model_file

model = fasttext.load_model(model_path)

df_list = []

for file in os.listdir(keywords_path):
	if "unstable" not in file and file.endswith(".csv"):
		filepath = os.path.join(keywords_path, file)
		df_list.append(pd.read_csv(filepath))

# Combine dataframes and collect unique (token, pred) pairs
df = pd.concat(df_list, ignore_index=True)[['token', 'pred']]

def save_embeddings(lang, df, model, output_path):
	rows = []
	for _, row in df.iterrows():
		token = str(row['token'])
		rows.append({
			"lang": lang,
			"text": token,
			"embed_last": model.get_word_vector(token).tolist(),
			"preds": eval(row['pred'])[0],
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

save_embeddings(lang, df, model, output_path)
print(f"Embeddings saved to {output_path}")
