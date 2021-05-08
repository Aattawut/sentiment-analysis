from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import nest_asyncio
from pyngrok import ngrok
import uvicorn

from nltk import NaiveBayesClassifier as nbc
from pythainlp.tokenize import word_tokenize
import codecs
from itertools import chain
import joblib
import pickle


app = FastAPI()
origins = ["*"]
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)


@app.get("/")
async def main():
    return {"message": "Hello World"}

mod = joblib.load('sentiment.model')
mod2 = joblib.load('vocabulary.model2')


@app.get("/api/predict_sentiment/{text}")
def predict_sentiment(text):
    test_sentence = str(text)
    featurized_test_sentence = {
        i: (i in word_tokenize(test_sentence.lower())) for i in mod2}
    classifier_2 = mod.classify(featurized_test_sentence)
    return {'results': classifier_2}


ngrok_tunnel = ngrok.connect(8000)
print('Public URL:', ngrok_tunnel.public_url)
nest_asyncio.apply()
uvicorn.run(app)

# if __name__ == "__main__":
#uvicorn.run(app, port=80)
