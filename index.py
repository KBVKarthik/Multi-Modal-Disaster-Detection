import pickle
import json
import os
import cv2
import requests
import wget
import re

import numpy as np

from flask import Flask, jsonify, request
from tensorflow.keras.models import load_model


encoded = {"Yes" : 1, "No" : 0}
decoded = {1 : "Yes", 0 : "No"}
SIZE = (224, 224)

class Logger:
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"

    @classmethod
    def log(cls, message, level=INFO):
        print(f">>> [{level}] {message}")

class TextPreProcessor:
    def __init__(self):
        pass

    def transform(self, text):
        text = text.lower()
        text = re.sub(r"http\S+", "", text)
        text = re.sub(r"@[^\s]+", "", text)
        text = re.sub(r"[^a-zA-Z\s]", "", text)

        text_features = []
        text_features.append(len(text))
        text_features.append(text.count("earthquake"))
        text_features.append(text.count("fire"))
        text_features.append(text.count("flood"))
        text_features.append(text.count("hurricane"))
        text_features.append(text.count("tornado"))
        text_features.append(text.count("volcano"))
        text_features.append(text.count("disaster"))
        text_features.append(text.count("emergency"))
        text_features.append(text.count("urgent"))
        text_features.append(text.count("help"))
        text_features.append(text.count("need"))
        text_features.append(text.count("please"))
        
        return text_features

class ImagePreProcessor:
    def __init__(self):
        pass
    
    def transform(self, images):
        pre_processed_images = []
        
        for image in images:
            i = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            i = cv2.resize(i, SIZE)
            i = i / 255.0

            pre_processed_images.append(i)
        
        return np.asarray(pre_processed_images)
            

class ImageClusteringModel:
    def __init__(self, load_from_path=None):     

        Logger.log("Loading Meta.json File") 
        meta = json.load(open(os.path.join(load_from_path, "meta.json"), "r"))
        self.feature_model = meta["base_model_name"]
        self.cluster_model = meta["clustering_model_name"]  

        Logger.log(f"Loading Base Model : {self.feature_model}")
        self.base = load_model(os.path.join(load_from_path, "base_model.h5"))

        Logger.log("Loading Feature Extractor Model")
        self.extractor = load_model(os.path.join(load_from_path, "extractor.h5"), compile=False)

        Logger.log(f"Loading Clustering Model : {self.cluster_model}")
        self.clustering = pickle.load(open(os.path.join(load_from_path, "clustering_model.pkl"), "rb"))

    
    def predict(self, X, verbose=0):
        features = self.extractor.predict(X, verbose=verbose)

        if self.cluster_model == "KMeans":
            clusters = self.clustering.predict(features)
        
        if self.cluster_model == "Agglomerative":
            clusters = self.clustering.fit_predict(features)

        return clusters.tolist()


image_clustering_model = ImageClusteringModel(load_from_path="./Models/InceptionV3 + KMeans")
text_pre_processor = TextPreProcessor()
image_pre_processor = ImagePreProcessor()

Logger.log(f"Loading XG-Boost Model")
XGB_model = pickle.load(open("./Models/XGB_Model.pkl", "rb"))

def fetch_tweet(id):
    bearer_token = "AAAAAAAAAAAAAAAAAAAAAA1tgwEAAAAAQoJkv4%2Ba9NYERkH0lxtu4s%2B3Me0%3D9TRxhuYQGJTNkTRyHBOeXVu1jyQElYS7Af8VsW9id3KNq3PN5K"
    
    def oauth(request):
        request.headers["Authorization"] = f"Bearer {bearer_token}"
        request.headers["User-Agent"] = "FYP"
        return request

    tweet_fields = ["id", "text", "context_annotations", "entities", "public_metrics", "attachments"]
    user_fields  = ["id", "name", "username", "created_at", "description", "entities", "location", "profile_image_url", "protected", "public_metrics", "url", "verified"]
    media_fields = ["url"]
    
    tweet_fields = "tweet.fields=" + ",".join(tweet_fields)
    user_fields  = "user.fields="  + ",".join(user_fields)
    media_fields = "media.fields=" + ",".join(media_fields)
    
    url = f"https://api.twitter.com/2/tweets?ids={id}&{tweet_fields}&{user_fields}&{media_fields}&expansions=attachments.media_keys,author_id"
    
    response = requests.request("GET", url, auth=oauth)
        
    if response.status_code != 200:
        raise Exception("Request returned an error: {} {}".format(response.status_code, response.text))
    
    data = response.json()

    extracted_data = {}
    extracted_data["text"] = data["data"][0]["text"]
    extracted_data["metrics"] = data["data"][0]["public_metrics"]
    extracted_data["user"] = {}
    extracted_data["user"]["description"] = data["includes"]["users"][0]["description"]
    extracted_data["user"]["description"] = data["includes"]["users"][0]["public_metrics"]
    extracted_data["images"] = [i["url"] for i in data["includes"]["media"] if i["type"] == "photo"]

    return extracted_data

def classify_image(images):
    Logger.log("Classifying Image")
    images = image_pre_processor.transform(images)
    clusters = image_clustering_model.predict(images)
   
    if np.all(clusters == 1):
        return "Relavent to a disaster"
    else:
        return "Not relavent to a disaster"

def classify_text(text):
    Logger.log("Classifying Text")
    text_features = text_pre_processor.transform(text)
    Logger.log(text_features)
    # prediction = XGB_model.predict([text_features])[0]

    prediction = 0
    
    if prediction == 0:
        return ["Not relevant to a disaster", "Not credible"]
    elif prediction == 1:
        return ["Relevant to a disaster", "Not credible"]
    elif prediction == 2:
        return ["Relevant to a disaster", "Credible"]

app = Flask(__name__)

@app.route('/', methods = ['POST'])
def index():
    if(request.method == 'POST'):
        tweet_URL = request.json["tweet-url"]

        tweet_data = fetch_tweet(tweet_URL.split("/")[-1])

        Logger.log(f"Downloading {len(tweet_data['images'])} images")
        image_files = [wget.download(i, "./Images", bar=None) for i in tweet_data["images"]]        
        images = [cv2.imread(i) for i in image_files]

        result = {
            "Image Result" : classify_image(images),
            "Text Result" : classify_text(tweet_data["text"]),
        }

        return jsonify(result)
    
if __name__ == '__main__':
    app.run(debug = True)