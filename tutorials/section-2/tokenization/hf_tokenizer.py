from transformers import BertTokenizer  # XLNetTokenizer, T5Tokenizer

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")  # xlnet-base-cased, t5-small

tokens = tokenizer.tokenize("I Love Data Science and Artificial Intelligence")

print(tokens)
