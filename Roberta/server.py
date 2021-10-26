# This file uses to deploy on a server
from flask import Flask, request
from sentence_transformers import SentenceTransformer, util


# Compute cosine-similarities
def sim(sent1, sent2):
    # load sentence similarity models
    model = SentenceTransformer('./all-roberta-large-v1')
    embedding1 = model.encode(sent1, convert_to_tensor=True)
    embedding2 = model.encode(sent2, convert_to_tensor=True)
    cosine_score = util.pytorch_cos_sim(embedding1, embedding2)
    return cosine_score.item()


app = Flask(__name__)
# app.config['ENV'] = 'development'
# app.config['DEBUG'] = True


@app.route('/auto_grading', methods=['GET', 'POST'])
def grading():
    if request.method == 'GET':
        student_answer = request.args['sa']
        reference_answer = request.args['ra']
    else:
        student_answer = request.form['sa']
        reference_answer = request.form['ra']

    cosine_similarity = sim(student_answer, reference_answer)
    if cosine_similarity < 0:
        cosine_similarity = 0.0000
    return {
        "student_answer": student_answer,
        "reference_answer": reference_answer,
        "similarity": cosine_similarity,
        "score": round(cosine_similarity * 100, 4)
    }


@app.route('/')
def home():
    print('[INFO from server.py] Access to home page successful.')
    return 'Hello, this is home page. I\'m running on docker now.'


if __name__ == '__main__':
    try:
        app.run(port=5000, debug=True)
    except RuntimeError as e:
        print('Mostly like an unexpected input has been sent to the server.')
