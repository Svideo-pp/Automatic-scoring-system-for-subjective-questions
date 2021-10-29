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
### Method 1 - Download all codes and dependencies, then build docker image on your own host machine
#### Step 1 - Download all files  this directory.
#### Step 2 - Download Huggingface pre-trained model from this link : https://huggingface.co/sentence-transformers/all-roberta-large-v1/tree/main .
You should archive all files from the link above into a signle dir and name it for example `all-roberta-large-v1` .
#### Step 3 - Move all files or dirs you downloaded into a url where you want to build your docker image. For instance `/path/to/auto-grading` .
#### Step 4 - To recap, by now you should have those files and dir structures as below:
auto-grading  
|_ _ _ _ all-roberta-large-v1  
|_ _ _ _ Dockerfile  
|_ _ _ _ Readme.md  
|_ _ _ _ gunicorn.conf.py  
|_ _ _ _ requirements.txt  
|_ _ _ _ server.py  
#### Step 5 - Open terminal and run the following codes
Change the path to where you want build your docker image, for example:
```
cd /path/to/auto-grading
```
Build the docker file, 'auto-grading-system' is the name of your image
```
sudo docker build -t 'auto-grading-system' .
```
This process will take for minutes if you run it first time.  
After finished, use this line to check your image
```
sudo docker image ls
```
To run it as a container, use
```
sudo docker run -it --rm -p 5000:5000 auto-grading-system
```

Congratulations, now you have finished all steps, and the auto-grading-system should run on the specificed address and port 🤗  
### Method 2 - Pull docker image from docker hub
Just run
```
sudo docker pull svideoier/auto-grading-system
```
This process will take for minutes if you run it first time since the whole image size is 5.69GB.  
After finished, use this line to check your image
```
sudo docker image ls
```
To run it as a container, use
```
sudo docker run -it --rm -p 5000:5000 auto-grading-system
```
Congratulations, now you have finished all steps, and the auto-grading-system should run on the specificed address and port 🤗  

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
