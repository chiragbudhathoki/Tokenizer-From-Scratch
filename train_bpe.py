import regex as re
from collections import defaultdict
def run_train_bpe(input_path:str,vocab_size:int,special_tokens:list[str]) -> tuple[dict[int,bytes],list[tuple[bytes,bytes]]]:
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    
    vocab:dict[int,bytes] = {i : bytes([i])for i in range(256) }

    next_id = 256
    for token_str in special_tokens:
        vocab[next_id] = token_str.encode('utf-8')
        next_id += 1

    with open(input_path,encoding = 'utf-8') as f:
        corpus = f.read()

    raw_bytes = [list(match.group().encode('utf-8')for match in re.finditer(PAT,corpus))]

    num_merges = vocab_size - len(vocab)
    merges:list[tuple[bytes,bytes]] = []


    for _ in range(num_merges):
        count = defaultdict(int)
        
        for chunks in raw_bytes:
            for i in range(len(chunks)-1):
                pairs = (chunks[i],chunks[i+1])
                count[pairs] += 1

        if not count:
            break
        best_pairs = max(count,key = count.get)

        pairs_byte = (vocab[best_pairs[0]],vocab[best_pairs[1]])
        merges.append(pairs_byte)
        vocab[next_id] = vocab[best_pairs[0]] + vocab[best_pairs[1]]
        new_raw_bytes = []
        
        for chunk in raw_bytes:
            i = 0
            new_chunk = []
            while i < len(chunk):
                if i < (len(chunk)-1) and (chunk[i],chunk[i+1]) == best_pairs:
                    new_chunk.append(next_id)
                    i += 2
                else:
                    new_chunk.append(chunk[i])
                    i += 1
            new_raw_bytes.append(new_chunk)
        
        raw_bytes = new_raw_bytes
        next_id += 1

    return vocab,merges

# import os
# def run_train_bpe(
#     input_path: str | os.PathLike,
#     vocab_size: int,
#     special_tokens: list[str],
#     **kwargs,
# ) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
#     """Given the path to an input corpus, run train a BPE tokenizer and
#     output its vocabulary and merges.

#     Args:
#         input_path (str | os.PathLike): Path to BPE tokenizer training data.
#         vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
#         special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
#             These strings will never be split into multiple tokens, and will always be
#             kept as a single token. If these special tokens occur in the `input_path`,
#             they are treated as any other string.

#     Returns:
#         tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
#             vocab:
#                 The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
#                 to bytes (token bytes)
#             merges:
#                 BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
#                 representing that <token1> was merged with <token2>.
#                 Merges are ordered by order of creation.
#     """






# import os

# # Create dummy training corpus
# test_filename = "dummy_corpus.txt"
# with open(test_filename, "w", encoding="utf-8") as f:
#     f.write("My name is Chirag Budhathoki. I am doing CS336")

# # Test train_bpe deliverable
# target_vocab_size = 260
# special_toks = ["<|endoftext|>"]

# vocab, merges = train_bpe(test_filename, vocab_size=target_vocab_size, special_tokens=special_toks)

# # CS336 Specification Assertions
# assert isinstance(vocab, dict), "vocab must be a dictionary!"
# assert all(isinstance(k, int) and isinstance(v, bytes) for k, v in vocab.items()), "vocab keys must be int and values bytes!"
# assert isinstance(merges, list), "merges must be a list!"
# if merges:
#     assert isinstance(merges[0], tuple), "merges items must be tuples!"
#     assert len(merges[0]) == 2 and isinstance(merges[0][0], bytes) and isinstance(merges[0][1], bytes), "merges item must be (bytes, bytes)!"

# print("Deliverable Validation Passed!")
# print(f"Vocab size: {len(vocab)}")
# print(f"Merges count: {len(merges)}")
# print(f"First 3 Merges: {merges[:3]}")

# # Clean up temporary file
# if os.path.exists(test_filename):
#     os.remove(test_filename)