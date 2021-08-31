import math
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow_hub as hub
from flask import Flask, request

module_url = "The path where you store the USE model in local."

# Load USE model from local
model = hub.load(module_url)
print ("module %s loaded" % module_url)

# Define a function that is easy to use the model
def embed(input):
  return model(input)


# deploy flask
app = Flask(__name__)
# app.config['ENV'] = 'development'
# app.config['DEBUG'] = True

@app.route('/auto_scoring', methods=['GET', 'POST'])
def scoring():
    if(request.method == 'GET'):
        student_answer = request.args['sa']
        reference_answer = request.args['ra']
    else:
        student_answer = request.form['sa']
        reference_answer = request.form['ra']
    student_answer_encode = embed([student_answer])
    reference_answer_encode = embed([reference_answer])
    cosine_similarities = tf.reduce_sum(tf.multiply(student_answer_encode, reference_answer_encode))
    clip_cosine_similarities = tf.clip_by_value(cosine_similarities, -1.0, 1.0)
    scores = 1.0 - tf.acos(clip_cosine_similarities).numpy() / math.pi
    res = scores * 100
    return {
        "student_answer": student_answer,
        "reference_answer": reference_answer,
        "similarity": scores,
        "score": res
    }

app.run(port=8080)
