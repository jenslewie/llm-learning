import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Section: Downloading and running an LLM
from transformers import AutoModelForCausalLM, AutoTokenizer

print("Loading microsoft/Phi-3-mini-4k-instruct model...")
model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Phi-3-mini-4k-instruct",
    device_map="mps",  # Use MPS for Apple Silicon instead of "cuda"
    torch_dtype="auto",
    local_files_only=True,
    attn_implementation="eager"
)
print("Loading microsoft/Phi-3-mini-4k-instruct tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    "microsoft/Phi-3-mini-4k-instruct",
    local_files_only=True
)

import torch

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

prompt = "Write an email apologizing to Sarah for the tragic gardening mishap. Explain how it happened.<|assistant|>"

print("Tokenizing the input prompt...")
inputs = tokenizer(prompt, return_tensors="pt").to(device)
input_ids=inputs['input_ids']

print("Printing the input...")
print(input_ids[0])
print(tokenizer.decode(input_ids[0]))

print("Generating the text...")
generation_output = model.generate(
    input_ids=input_ids,
    attention_mask=inputs['attention_mask'],
    max_new_tokens=20
)

print("Printing the output...")
print(generation_output[0])
print(tokenizer.decode(generation_output[0]))

# Section: LLM Tokenization - Comparing Trained LLM Tokenizers
text = """
English and CAPITALIZATION
🎵 鸟
show_tokens False None elif == >= else: two tabs:"    " Three tabs: "       "
12.0*50=600
"""

from transformers import AutoTokenizer

colors_list = [
    '102;194;165', '252;141;98', '141;160;203',
    '231;138;195', '166;216;84', '255;217;47'
]

def show_tokens(sentence, tokenizer_name):
    print(f"\n\n========== Loading: {tokenizer_name} ==========")
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        return

    # --- 1. Metadata (Class & Algorithm) ---
    cls_name = type(tokenizer).__name__

    # Attempt to detect algorithm for Fast tokenizers
    if tokenizer.is_fast and hasattr(tokenizer, "backend_tokenizer"):
        algo_type = type(tokenizer.backend_tokenizer.model).__name__
    else:
        algo_type = "Standard (Python/Slow)"

    # --- 2. Gather Data ---
    all_special_ids = set(tokenizer.all_special_ids)
    added_vocab_map = tokenizer.get_added_vocab()

    # Filter 'added_vocab' to find tokens that are NOT in 'all_special_ids'
    # These are specific merges or custom tokens not mapped to roles like [CLS]
    additional_added_tokens = {
        k: v for k, v in added_vocab_map.items()
        if v not in all_special_ids
    }

    # --- 3. Print Overview ---
    print(f"Class:          {cls_name}")
    print(f"Algorithm:      {algo_type}")
    print(f"Vocab Size:     {tokenizer.vocab_size}")

    print(f"\n[Token Statistics]")
    print(f"Total Special Tokens:   {len(all_special_ids)}")
    print(f"Total Added Tokens:     {len(added_vocab_map)}")
    print(f"  L Purely Additional:  {len(additional_added_tokens)} (Excluded from Special Roles)")

    # --- 4. Special Token Roles ---
    if tokenizer.special_tokens_map:
        print(f"\n[Special Token Roles]")
        print(f"{'Role':<25} | {'Token':<30} | {'ID'}")
        print("-" * 70)

        for role, value in tokenizer.special_tokens_map.items():
            # Handle list values (e.g. multiple additional_special_tokens)
            tokens_to_print = value if isinstance(value, list) else [value]

            for t in tokens_to_print:
                # Safe conversion in case a string in the map isn't in vocab
                t_id = tokenizer.convert_tokens_to_ids(
                    t) if t in tokenizer.get_vocab() or t in added_vocab_map else "N/A"
                print(f"{role:<25} | {str(t):<30} | {t_id}")
    else:
        print("(No special tokens map found)")

    # --- 5. Additional Added Vocab (Filtered) ---
    if additional_added_tokens:
        print(f"\n[Additional Added Tokens] (Excluding Special Tokens)")

        # Sort by ID for cleaner viewing
        sorted_additional = sorted(additional_added_tokens.items(), key=lambda item: item[1])

        print(f"{'Token':<25} | {'ID'}")
        print("-" * 40)
        for token, t_id in sorted_additional:
            print(f"{token:<25} | ID: {t_id}")
    else:
        print(f"\n[Additional Added Tokens]")
        print("None. (All added tokens are currently classified as Special Tokens).")

    token_ids = tokenizer(sentence).input_ids
    print(f"\n[Token Usage]")
    print(f"Token Count:           {len(token_ids)}")
    print(f"Distinct Token Count:  {len(set(token_ids))}\n")
    for idx, t in enumerate(token_ids):
        print(
            f'\x1b[0;30;48;2;{colors_list[idx % len(colors_list)]}m' +
            tokenizer.decode(t) +
            '\x1b[0m',
            end=' '
        )

show_tokens(text, "bert-base-uncased")
show_tokens(text, "bert-base-cased")
show_tokens(text, "google/flan-t5-small")
show_tokens(text, "gpt2")
show_tokens(text, "Xenova/gpt-4")
show_tokens(text, "bigcode/starcoder2-15b")
show_tokens(text, "facebook/galactica-1.3b")
show_tokens(text, "microsoft/Phi-3-mini-4k-instruct")

# Section: Creating Contextualized Word Embeddings with Language Models
from transformers import AutoModel, AutoTokenizer

print("\n\nLoading microsoft/deberta-base tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-base")

print("Loading microsoft/deberta-v3-xsmall model...")
model = AutoModel.from_pretrained("microsoft/deberta-v3-xsmall")

print("Tokenizing the input prompt...")
tokens = tokenizer('Hello world', return_tensors='pt')

print("Processing the tokens...")
output = model(**tokens)[0]

print("Printing the output...")
print(output.shape)
for token in tokens['input_ids'][0]:
    print(tokenizer.decode(token))
print(output)

# Section: Text Embeddings (For Sentences and Whole Documents)
from sentence_transformers import SentenceTransformer

print("\n\nLoading sentence-transformers/all-mpnet-base-v2 model...")
model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

print("Converting text to text embeddings...")
vector = model.encode("Best movie ever!")

print(vector.shape)

# Section: Word Embeddings Beyond LLMs
import gensim.downloader as api

# Download embeddings (66MB, glove, trained on wikipedia, vector size: 50)
# Other options include "word2vec-google-news-300"
# More options at https://github.com/RaRe-Technologies/gensim-data
print("\n\nLoading glove-wiki-gigaword-50 model...")
model = api.load("glove-wiki-gigaword-50")

print("10 most similar tokens")
print(model.most_similar([model['king']], topn=11))

# Section: Training a Song Embedding Model
import pandas as pd
from urllib import request

print("\n\nGetting the playlist dataset file...")
data = request.urlopen('https://storage.googleapis.com/maps-premium/dataset/yes_complete/train.txt')

print("Parsing the playlist dataset file. Skip the first two lines as they only contain metadata...")
lines = data.read().decode("utf-8").split('\n')[2:]

print("Removing playlists with only one song...")
playlists = [s.rstrip().split() for s in lines if len(s.split()) > 1]

print("Loading song metadata...")
songs_file = request.urlopen('https://storage.googleapis.com/maps-premium/dataset/yes_complete/song_hash.txt')
songs_file = songs_file.read().decode("utf-8").split('\n')
songs = [s.rstrip().split('\t') for s in songs_file]
songs_df = pd.DataFrame(data=songs, columns = ['id', 'title', 'artist'])
songs_df = songs_df.set_index('id')

print("Printing paylists...")
print( 'Playlist #1:\n ', playlists[0], '\n')
print( 'Playlist #2:\n ', playlists[1])

print("Printing songs...")
print( '  Songs count: ', len(songs))

from gensim.models import Word2Vec

print("Training our Word2Vec model...")
model = Word2Vec(
    playlists, vector_size=32, window=20, negative=50, min_count=1, workers=4
)

song_id = 2172

# Ask the model for songs similar to song #2172
model.wv.most_similar(positive=str(song_id))

print("Song id: ", song_id)
print(songs_df.iloc[song_id])

import numpy as np

def print_recommendations(_song_id):
    similar_songs = np.array(
        model.wv.most_similar(positive=str(_song_id),topn=10)
    )[:,0]
    print(songs_df.iloc[similar_songs])

# Extract recommendations
print_recommendations(song_id)