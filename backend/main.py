import string
import re
from typing import List
import nltk
from nltk.stem import *
from nltk.corpus import stopwords 
from pydantic import BaseModel
from fastapi import FastAPI
import pickle


def transform_text(text):
  text = text.lower()
  text = nltk.word_tokenize(text)
  ps = PorterStemmer()
  y=[]
  for i in text:
    if i.isalnum():
      y.append(i)
  text = y[:]
  y.clear()
  for i in text:
    if i not in stopwords.words('english') and i not in string.punctuation:
      y.append(i)
  text = y[:]
  y.clear()
  for i in text:
    y.append(ps.stem(i))

  return " ".join(y)

def getUrlLinks(url):
  url_extract_pattern = "https?:\\/\\/(?:www\\.)?[-a-zA-Z0-9@:%._\\+~#=]{1,256}\\.[a-zA-Z0-9()]{1,6}\\b(?:[-a-zA-Z0-9()@:%_\\+.~#?&\\/=]*)"
  return re.findall(url_extract_pattern, url)



model=pickle.load(open('model.pkl','rb'))
tf=pickle.load(open('vectorizer.pkl','rb'))


class Query(BaseModel):
    # message: str
    messages: List[str]=[]


app = FastAPI()

@app.get("/")
async def root(query:Query):
  # message_transformed=transform_text(query.message)
  # message_vector=tf.transform([message_transformed])
  # pred = model.predict(message_vector)[0]

  # if(pred==1):
  #   return {"spam": 1}
  # else:
  #   return {"spam": 0}
      response_data = []
      for message in query.messages:
            urls = getUrlLinks(message)

            message_transformed=transform_text(message)
            message_vector=tf.transform([message_transformed])
            pred = model.predict(message_vector)[0]

            msg_data = {}
            
            response_data.append(msg_data)
            
            url_data = []
            for url in urls:
                  url_json ={}
                  url_json["url"] = url
                  url_json["spam"] = 1
                  url_data.append(url_json)
            
            if pred==1:
                  msg_data = {"message":message, "spam":1, "urls":url_data}
            else:
                  msg_data = {"message":message, "spam":0, "urls":url_data}

      response = {"result":response_data}
      return response



