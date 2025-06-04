import sys
import os
import gc
import pandas as pd
from collections import defaultdict
from ast import literal_eval
import datasets

# To make accessing path arguments easier
arg_keys = ['hf_dir', 'keyword_dir', 'result_dir', 'lang']
args = dict(zip(arg_keys, sys.argv[1:]))

hf_path = os.path.join(args['hf_dir'], f"{args['lang']}.hf")
kw_path = os.path.join(args['keyword_dir'], f"{args['lang']}")
hf_out = os.path.join(args['result_dir'], f"data-with-keywords-{args['lang']}")
os.makedirs(args['result_dir'], exist_ok=True)

# We concatenate the kw files for all registers into a single dataframe
df_list = []
for file in os.listdir(kw_path):
	if file.endswith("csv") and "unstable" not in file:
		df_list.append(pd.read_csv(os.path.join(kw_path, file)))
kw = pd.concat(df_list).reset_index(drop=True)

# Memory stuff
del df_list
gc.collect()

# Transform the predictions, labels, and IDs into lists
cols_to_transform = ['pred', 'label', 'id']
kw[cols_to_transform] = kw[cols_to_transform].map(literal_eval)
# Set of all IDs in the keyword data
id_set = kw['id'].explode().unique()
# The original document text data resides here
data = datasets.load_from_disk(hf_path)

# The dataset is filtered to only include the IDs present in the keywords
data = data.filter(lambda ex: ex['id'] in id_set)

# Here we basically reverse-map the keyword data
# FROM:	Token-register pair --> Many text IDs
# TO:	Text ID --> Many token-register pairs
# Also, unnecessary fields are implicitly dropped at this step
flattened = []
for _, row in kw.iterrows():
	token, pred, ids = row['token'], row['pred'], row['id']
	for id_ in ids:
		flattened.append({'id': id_, 'token': token, 'pred': pred})

flat_kw = pd.DataFrame(flattened)

def build_keywords(group):
	"""
    This helper takes a a single-ID group of keywords and builds a list of
	dictionaries, each containing a token and its associated labels for that ID.
	"""
	token_to_preds = defaultdict(set)
	for _, row in group.iterrows():
		for label in row['pred']:
			token_to_preds[row['token']].add(label)
	return [
		{'token': token, 'preds': sorted(list(preds))}
		for token, preds
		in token_to_preds.items()
	]

# We group the keywords by ID and apply the build_keywords function to each
# group. This will create a dictionary where the keys are IDs and the values are
# lists of keywords. Each keyword is represented as a dictionary with 'token'
# and 'preds' keys. The 'preds' key contains a list of labels associated with
# that token.

keywords_by_id = (
	flat_kw.groupby('id', group_keys=False)
	.apply(build_keywords, include_groups=False)
	.to_dict()
)

# Add the freshly built keywords and their predictions to the dataset
data = data.map(lambda x: {'keywords': keywords_by_id.get(x['id'], [])})

data.save_to_disk(hf_out)
