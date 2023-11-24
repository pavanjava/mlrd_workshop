from transformers import pipeline

# Creating a Text2TextGenerationPipeline for language translation
pipe = pipeline(task='text2text-generation', model='facebook/m2m100_418M')

# Converting
response = pipe("That is a flower", forced_bos_token_id=pipe.tokenizer.get_lang_id(lang='hi'))

if response is not None:
    print(response[0])