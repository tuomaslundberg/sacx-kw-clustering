import sys
import os
import csv
import ast
import numpy as np
from collections import defaultdict

# To make accessing path arguments easier
arg_keys = ['input_dir', 'output_dir', 'lang']
args = dict(zip(arg_keys, sys.argv[1:]))

input_dir = os.path.join(args['input_dir'], args['lang'])
output_path = os.path.join(args['output_dir'], f"avg-embeddings_{args['lang']}.tsv")
os.makedirs(args['output_dir'], exist_ok=True)

# --- Load all folds ---
num_folds = 10
fold_files = [os.path.join(input_dir, f"embeddings-fold_{i + 1}.tsv") for i in range(num_folds)]

agg = defaultdict(list)            # (token, pred) -> list of embeddings
seen_in_folds = defaultdict(set)   # (token, pred) -> set of fold indices

for fold_idx, fpath in enumerate(fold_files):
    with open(fpath, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            token = row["text"]
            pred = row["preds"]

            # Parse embedding string into numpy array
			# embeddings are stored like "[0.123, 0.456, ...]"
            try:
                embed = np.array(ast.literal_eval(row["embed_last"]), dtype=np.float32)
            except Exception as e:
                raise ValueError(
                    f"Failed parsing embedding for {token}/{pred} in {fpath}: {row['embed_last']}"
                ) from e

            agg[(token, pred)].append(embed)
            seen_in_folds[(token, pred)].add(fold_idx)

# --- Average per keyword-prediction pair ---
final_rows = []
warnings = []

for (token, pred), embeds in agg.items():
    stacked = np.stack(embeds, axis=0)  # [n_folds_seen, dim]
    mean_embed = stacked.mean(axis=0)
    final_rows.append({
        "lang": args["lang"],
        "text": token,
        "embed_last": mean_embed.tolist(),
        "preds": pred
    })

    # Check if missing from some folds
    if len(seen_in_folds[(token, pred)]) < num_folds:
        missing = set(range(num_folds)) - seen_in_folds[(token, pred)]
        warnings.append(f"{token}/{pred} missing from folds {sorted(missing)}")

# --- Write out final TSV ---
with open(output_path, "w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["lang", "text", "embed_last", "preds"], delimiter="\t")
    writer.writeheader()
    for row in final_rows:
        writer.writerow({
            "lang": row["lang"],
            "text": row["text"],
            "embed_last": str(row["embed_last"]),
            "preds": row["preds"]
        })

print(f"✅ Averaged embeddings saved to {output_path}")
if warnings:
    print("\n⚠️ Inconsistencies detected:")
    for w in warnings:
        print(" -", w)

