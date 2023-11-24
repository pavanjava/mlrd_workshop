from transformers import pipeline

# Creating a Text2TextGenerationPipeline for language translation
pipe = pipeline(model="finiteautomata/bertweet-base-sentiment-analysis")

# data
data = ["I love Data Science, AI and Generative AI", "last week i saw a movie which is worst and waste of time"]

# Converting
response = pipe(data)

if response is not None:
    print(response)