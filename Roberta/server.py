# This file uses to deploy on a server

from flask import Flask, request
from sentence_transformers import SentenceTransformer, util


# load sentence similarity models
model = SentenceTransformer('C:/Users/svideo/Desktop/HuggingFace Model/all-roberta-large-v1')
print('model all_roberta_large_v1 loaded')


#Compute cosine-similarits
def sim(sent1, sent2):
    embedding1 = model.encode(sent1, convert_to_tensor=True)
    embedding2 = model.encode(sent2, convert_to_tensor=True)
    cosine_score = util.pytorch_cos_sim(embedding1, embedding2)
    return cosine_score.item()


app = Flask(__name__)
# app.config['ENV'] = 'development'
# app.config['DEBUG'] = True

@app.route('/auto_scoring', methods=['GET', 'POST'])
def scoring():
    if (request.method == 'GET'):
        student_answer = request.args['sa']
        reference_answer = request.args['ra']
    else:
        student_answer = request.form['sa']
        reference_answer = request.form['ra']

    cosine_similarity = sim(student_answer, reference_answer)
    return {
        "student_answer": student_answer,
        "reference_answer": reference_answer,
        "similarity": cosine_similarity,
        "score": round(cosine_similarity * 100, 4)
    }


try:
    app.run(port=8080)
except:
    print("running error")

