# Roberta model
Roberta is the abbreviation of Robustly Optimized BERT Pretraining Approach.

View on the github: https://github.com/pytorch/fairseq

Download from huggingface: https://huggingface.co/sentence-transformers/all-roberta-large-v1

## Overview
Roberta is an improved model based on BERT. The main improvements include the following four aspects.
1. Changing the static masking into dynamic masking
2. Cancel the Next Sentence Prediction (NSP) task in pretraining process and replace it by FULL-SENTENCES
3. Using larger batches, datasets, and pretraining with more steps.
4. Using Byte-Pair Encoding (BPE) with more subword units than BERT.

Due to the modifications mentioned above. The Roberta model achieves state-of-the-art results on all 9 of the GLUE task development sets. Specifically, as for the STS task, it has got a Pearson correlation coefficient of 92.4 from Liu et al. (2019).

see more about GLUE task and STS-B: https://openreview.net/pdf?id=rJ4km2R5t7

## Usage
Download from huggingface: https://huggingface.co/sentence-transformers/all-roberta-large-v1

You can also download it from the google drive below：
https://drive.google.com/drive/folders/1JkoWtp5gJgoLLjlwh4ioLBdzla9SeAx8?usp=sharing

Then you need to change the `model_url` variable in the 8th line of the `server.py` file to the path where you store the downloaded model in your local computer.
A sample is like:
```python
module_url = "C:\\Users\\svideo\\Desktop\\HuggingFace Model\\all-roberta-large-v1"
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
