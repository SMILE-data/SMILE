# SMILE: A Multi-Modal Dataset for Understanding Laughter

This is the repository of SMILE: A Multimodal Dataset for Understanding Laughter. 
It comprises SMILE dataset, and codes involving processing dataset and generation explanation reason behind the laughter.

## Installation
```
$ conda create -n SMILE python==3.10.11
$ conda activate SMILE

# move to fastchat/ directory
$ pip3 install --upgrade pip  
$ pip3 install -e .
$ pip3 install openai
```


### Download the SMILE Dataset
1. Download SMILE dataset in [here](https://drive.google.com/file/d/17v318r5DQp4loh28WD9vVUtvS5xeKpR9/view)
2. unzip midifiles(`commu_midi.tar`).
    ```
    $ cd ComMU-Code
    $ tar -xvf /dataset/commu_midi.tar -C /dataset/
    ```
    and if the project tree looks like this, it is ready for preprocessing. 
    ```
    .
    ├── commu_meta.csv
    └── commu_midi
        └── train
            └── raw
                └── midifiles(.mid)
        └── val
            └── raw
                └── midifiles(.mid)
    ``` 

## Data processing



## Evaluation

### Laugh reasoning 
We provide the inference code for in-context and zero-shot experiment using GPT3. 
As the fine-tuneded GPT3 requires a certain openai api-key which the model was fine-tuned on, we instead provide the inferecne code for fine-tuned model using LLaMA. 

#### In-context and Zero-shot experiment (GPT3)
Note that running GPT3 requires your own openai api-key and also charges.
Replace the { } with your own information. 
```
$ python gpt3_inferece.py -openai_key {your openai api key} -engine {gpt3 model} -shot {fewshot or zeroshot} -val_data {path/for/validation_data} -train_data {path/for/validation_data} 
```

