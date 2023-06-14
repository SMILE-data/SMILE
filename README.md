# SMILE: A Multi-Modal Dataset for Understanding Laughter

This is the repository of SMILE: A Multimodal Dataset for Understanding Laughter. 
It comprises SMILE dataset, and codes involving processing dataset and generation explanation reason behind the laughter.

## Installation
```
$ conda create -n SMILE python==3.10.11
$ conda activate SMILE

# move to FastChat/ directory
$ cd FastChat

$ pip3 install --upgrade pip  
$ pip3 install -e .
$ pip3 install openai
$ pip3 install scikit-image
$ pip3 install evaluate
$ pip3 install bert-score
```


### Download the SMILE Dataset
1. Download SMILE dataset in [here](https://drive.google.com/file/d/17v318r5DQp4loh28WD9vVUtvS5xeKpR9/view)
2. unzip the dataset.

    ```
    
    ├── annotations
    |    ├── data_split.json
    |    ├── GT_laughter_reason.json
    |    └── multimodal_textual_representation.json
    |
    └── videos
         └── SMILE_videos.zip
                ├── video_clips
                └── video_segments
    
    ``` 
3. Details about each file
   - data_split.json: key index for train, validation, test split
   - GT_laughter_reason.json: Ground-Truth laughter reason for video clip
   - multimodal_textual_representation.json: multimodal textual representation encoded from video clip
   - video_clips: 887 video clips from sitcom and TED, **Note**: sitcom has an underbar in the key index, while TED does not.
   - video_segments: 4482 video segments trimmed from video clip by utterances.



## Evaluation

### Laugh reasoning 
We provide the inference code for in-context and zero-shot experiment using GPT3. 
As the fine-tuneded GPT3 requires a certain openai api-key which the model was fine-tuned on, we instead provide the inferecne code for fine-tuned model using LLaMA. 

#### In-context and Zero-shot experiment (GPT3)
Note that running GPT3 requires your own openai api-key and also charges.
Replace the { } with your own information. 
```
$ python gpt3_inferece.py -openai_key {your openai api key} -engine {name of gpt3 model} -shot {fewshot or zeroshot} -val_data {path/for/validation_data} -train_data {path/for/train_data} 
```
#### Fine-tuned experiment (LLaMA)
| Training data | Link |
|-------------|------|
| SMILE       | 테스트2 |
| SMILE_TEL   | 테스트2 |
| 테스트1        | 테스트2 |

Replace the { } with your own information. 
```
$ python FastChat/fastchat/serve/inference.py -model_path {path/for/fine-tuned model} -val_data {path/for/validation_data} -train_data {path/for/train_data} 
```
