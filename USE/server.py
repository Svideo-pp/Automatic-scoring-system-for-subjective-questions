import math
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow_hub as hub
from flask import Flask, request

module_url = "C:\\Users\\svideo\\AppData\\Local\\Temp\\tfhub_modules\\063d866c06683311b44b4992fd46003be952409c"
model = hub.load(module_url)
print ("module %s loaded" % module_url)
def embed(input):
  return model(input)

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


app.run(port=80)
# # What functions do operators perform in programming?
# student_answer1 = 'Operator is a special symbol used to operate a data value.'
# student_answer2 = 'Function to operate a data.'
# student_answer3 = 'Operators are used to manipulating or performing calculations on a variable value.'
# student_answer4 = 'operator function = manage all forms of programming.'
# reference_answer = 'Operators function to manipulate the value of a variable.'
#
# print(scoring(student_answer1, reference_answer))
# print(scoring(student_answer2, reference_answer))
# print(scoring(student_answer3, reference_answer))
# print(scoring(student_answer4, reference_answer))

# 127.0.0.1:8080/auto_scoring?sa=Operator is a special symbol used to operate a data value.&ra=Operators function to manipulate the value of a variable.
# gunicorn start:app -c gunicorn.conf.py