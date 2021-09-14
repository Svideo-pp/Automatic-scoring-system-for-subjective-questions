# Roberta model
Roberta is the abbreviation of Robustly Optimized BERT Pretraining Approach.

View on the github: https://github.com/pytorch/fairseq

Download from huggingface: https://huggingface.co/sentence-transformers/all-roberta-large-v1

## Overview
This...................................... This model can achieve a Pearson correlation coefficient of 0.8036394664851696 on STS-Benchmark.

see more about GLUE task and STS-B: https://openreview.net/pdf?id=rJ4km2R5t7

## Usage
You can also download it from the google drive below：
https://drive.google.com/drive/folders/1JkoWtp5gJgoLLjlwh4ioLBdzla9SeAx8?usp=sharing

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
# The result of all_roberta_large_v1 :
# The reference answer is:  Operators function to manipulate the value of a variable.
# Operator is a special symbol used to operate a data value. 		                         Score: 0.7710
# Function to operate a data. 		                                                     Score: 0.5999
# Operators are used to manipulating or performing calculations on a variable value. 		 Score: 0.8681
# operator function = manage all forms of programming. 		                             Score: 0.6263
# Operators function to manipulate the address of a variable. 		                     Score: 0.7348
```


## Reference
[1] Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike Lewis, Luke Zettlemoyer, Veselin Stoyanov. Roberta: A Robustly Optimized BERT Pretraining Approach. arXiv:1907.11692, 2019.
