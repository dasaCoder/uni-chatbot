import pandas as pd
DATA_URL = "data/data.csv"

dataset = pd.read_csv(DATA_URL)
##dataset.head()

from flask import Flask, jsonify
from flask import request, abort

from flask_cors import CORS, cross_origin

app = Flask(__name__)

cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'


@app.route('/api/v1/predict', methods=['POST'])
def create_task():
    if not request.json or not 'text' in request.json:
        abort(400)
    
    print(request.json)
    return jsonify(predict(request.json['text']))



from sklearn.naive_bayes import MultinomialNB
from sklearn import tree
from sklearn.feature_extraction.text import CountVectorizer

#classifier = MultinomialNB()
classifier = tree.DecisionTreeClassifier()
stop_words = ['i','please','a','to']

countervect = CountVectorizer(min_df=5)  ##,stop_words=stop_words
vectorized_text = pd.DataFrame(countervect.fit_transform(dataset['text']).toarray(),columns=countervect.get_feature_names(), index=None)

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(vectorized_text,dataset['class'],test_size=0.1)

classifier.fit(vectorized_text,dataset['class'])

classifier.score(X_train,y_train)

## prepare text for the classification

def prepare_text(text):
  ##t = simplify_sinhalese_text(text)
  return countervect.transform([text]).toarray()

def predict(text):

  guess_proba = classifier.predict_proba(prepare_text(text))
  max_value = max(guess_proba[0])
  print(max_value)
  if max_value > 0.4:
    return {'result': classifier.predict(prepare_text(text))[0]}
  else: 
    return {'result' : "NOT_FOUND"}

##predict("hi, good morning")

if __name__ == '__main__':
    app.run(debug=True)

