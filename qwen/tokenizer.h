// Qwen2 / Qwen2.5 byte-level BPE tokenizer: the contract between qwen/tokenizer.c and qwen/qwen.c.
// The data file is written by tools/qwen_tok_export.py from Hugging Face's tokenizer.json.
#pragma once

typedef struct Tokenizer Tokenizer;

Tokenizer *tok_load(const char *path); // NULL (with a message on stderr) on failure
void tok_free(Tokenizer *t);

// Encode UTF-8 text to token ids exactly as Hugging Face `tokenizers` does for Qwen2.5 (pre-tokenizer regex,
// byte-to-unicode mapping, BPE merges by rank). With allow_special != 0, added tokens such as "<|im_start|>"
// appearing in the text become their single ids; otherwise they are encoded as plain text.
// Returns the number of ids written, or -1 if more than max_ids would be needed.
int tok_encode(Tokenizer *t, const char *text, int allow_special, int *ids, int max_ids);

// Raw bytes of one token (not NUL-terminated; a multi-byte UTF-8 character may span tokens). *len gets the length.
const char *tok_decode(Tokenizer *t, int id, int *len);

int tok_id(Tokenizer *t, const char *added_token); // id of an added token like "<|im_end|>", or -1
int tok_vocab_size(Tokenizer *t);                  // number of ids the tokenizer knows (151665 for Qwen2.5)
