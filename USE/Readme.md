# USE model
USE is the abbreviation of Universal Sentence Encoder. 

View on the github: https://github.com/tensorflow/hub/blob/master/examples/colab/semantic_similarity_with_tf_hub_universal_encoder.ipynb

Download from tensroflow_hub: https://tfhub.dev/google/universal-sentence-encoder/4

## Overview
The model is trained and optimized for greater-than-word length text, such as sentences, phrases or short paragraphs. It is trained on a variety of data sources and a variety of tasks with the aim of dynamically accommodating a wide variety of natural language understanding tasks. The input is variable length English text and the output is a 512 dimensional vector. This model can achieve a Pearson correlation coefficient of 0.8036394664851696 on STS-Benchmark.

see more about GLUE task and STS-B: https://openreview.net/pdf?id=rJ4km2R5t7

## Usage
First download all files in the folder. It is important to note that because the USE model is too large to upload to github, you need to download it from the tensorflow-hub link mentioned above. You can also download it from the google drive below：


Then you need to change the `model_url` variable in the 7th line of the `server.py` file to the path where you store the downloaded model in your local computer.
A sample is like:
```python
module_url = "C:\\Users\\svideo\\AppData\\Local\\Temp\\tfhub_modules\\063d866c06683311b44b4992fd46003be952409c"
```

After that, you need finish the parameter `WORKDIR` of Dockerfile, it is the path that the model to be deployed on the server.
One instance is like:
```
WORKDIR /Project/demo
```

Congratulations, all the preparations before building the docker image are completed. 🤗

## Examples
```python
# Question: What functions do operators perform in programming?
student_answer1 = 'Operator is a special symbol used to operate a data value.' 
student_answer2 = 'Function to operate a data.'
student_answer3 = 'Operators are used to manipulating or performing calculations on a variable value.'
student_answer4 = 'operator function = manage all forms of programming.'
reference_answer = 'Operators function to manipulate the value of a variable.'

# Sentence vector of student_answers' cosine similarity with reference_answer
# answer1: 0.6502556480355459
# answer2: 0.6240073069461776
# answer3: 0.7578465712798699
# answer4: 0.6884413384103176

# score:
# answer1: 65.02556480355459
# answer2: 62.40073069461776
# answer3: 75.78465712798699
# answer4: 68.84413384103176
```


## Reference
[1] Daniel Cer, Yinfei Yang, Sheng-yi Kong, Nan Hua, Nicole Limtiaco, Rhomni St. John, Noah Constant, Mario Guajardo-Céspedes, Steve Yuan, Chris Tar, Yun-Hsuan Sung, Brian Strope, Ray Kurzweil. Universal Sentence Encoder. arXiv:1803.11175, 2018.
