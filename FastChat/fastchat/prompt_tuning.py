import json
import pdb
import pickle
# with open("/home/hyun/project/LLM/FastChat/playground/data/ted/ted_reasoning_val.json", "r") as f:
#     sitcom_reasoning = json.load(f)
#
# with open("/home/hyun/project/LLM/FastChat/playground/ours_dataset/mustard_att_mturk.json", "r") as f:
#     sitcom_attr_mturk = json.load(f)

with open("/home/hyun/project/LLM/FastChat/playground/ours_dataset/ted_gt_temp05.json", "r") as f:
    sitcom_gt = json.load(f)

with open("/home/hyun/project/LLM/FastChat/playground/ours_dataset/urfunny_attr_all.json", "r") as f:
    sitcom_attr_all = json.load(f)
#
# with open("/home/hyun/project/LLM/FastChat/playground/ours_dataset/sarcasm_data.json", "r") as f:
#     sarcasm_data = json.load(f)


with open("/home/hyun/project/LLM/FastChat/playground/ours_dataset/split_indices.p", 'rb') as f:
    u = pickle._Unpickler(f)
    u.encoding = 'latin1'
    sitcom_train_val_idx = u.load()

# with open("/home/hyun/project/LLM/FastChat/playground/data/sitcom/sitcom_detection_train_1.json", "r") as f:
#     detect_valid = json.load(f)


# Reasoning
with open("/home/hyun/project/LLM/FastChat/playground/data/ted/ted_reasoning_train.json", "r") as f:
    sitcom_reasoning_train = json.load(f)

with open("/home/hyun/project/LLM/FastChat/playground/data/ted/ted_reasoning_val.json", "r") as f:
    sitcom_reasoning_val = json.load(f)


train_key = [i['id'] for i in sitcom_reasoning_train]
val_key = [i['id'] for i in sitcom_reasoning_val]

#####FOR FULL TRAINING
# sitcom_reasoning = []
# for idx, key in enumerate(val_key):
#     sitcom_attr_sample = sitcom_attr_all[key]
#     sitcom_gt_sample = sitcom_gt[key]
#     prompt = f"Reasoning task: you are going to answer why the audience laughed given the video clip. The video clip from TED, titled {sitcom_attr_sample['Video title']} is given with multimodal information(Utterance, Facial Action Units, Video caption, Acoustic features(6 dimension; 1.mean of F0 contour, 2.var of F0 contour, 3. mean of energy contour, 4. var of energy contour, 5. jitter, 6. shimmer)):"
#     question= "Give a detailed reason why the audience laugh, marked as (audience laughed), at most 30 words, starting with `the audience laughed because'"
#     input_attr = sitcom_attr_sample["Video clips"]
#     inputs = prompt + "Given video clip:" + str(input_attr) + question
#     sitcom_reasoning.append({'id': f"{key}", "conversations": [{'from': "human", "value": inputs}, {'from': "gpt", "value": sitcom_gt_sample}]})
#
# # save new reasoning data
# with open("/local_data2/sung/dataset/ted_reasoning_val_v2.json", "w") as f:
#     json.dump(sitcom_reasoning, f, indent=2)

# sitcom_reasoning = []
# for idx, key in enumerate(train_key):
#     sitcom_attr_sample = sitcom_attr_all[key]
#     sitcom_gt_sample = sitcom_gt[key]
#     prompt = f"Reasoning task: you are going to answer why the audience laughed given the video clip. The video clip from TED, titled {sitcom_attr_sample['Video title']}, with multimodal information(Utterance, Facial Action Units, Video caption, Acoustic features(6 dimension; 1.mean of F0 contour, 2.var of F0 contour, 3. mean of energy contour, 4. var of energy contour, 5. jitter, 6. shimmer)) is given. The audience laughing moment is makred as (audience laughing) in certain utterance. Explain why the audience laughed given the conversation, starting wtih 'The audience laughed because '"
#     input_attr = sitcom_attr_sample["Video clips"]
#     inputs = prompt + " video clip:" + str(input_attr)
#     sitcom_reasoning.append({'id': f"{key}", "conversations": [{'from': "human", "value": inputs}, {'from': "gpt", "value": sitcom_gt_sample}]})
#
#
# # save new reasoning data
# with open("/local_data2/sung/dataset/ted_reasoning_train.json", "w") as f:
#     json.dump(sitcom_reasoning, f, indent=2)


#######FOR TRANSCRIPT ONLY TRAINING
# sitcom_reasoning = []
# for idx, key in enumerate(train_key):
#     sitcom_attr_sample = sitcom_attr_all[key]
#     sitcom_gt_sample = sitcom_gt[key]
#
#     prompt = f"The video clip from TED titled {sitcom_attr_sample['Video title']} is given with utterances for each clip. The audience laughing moment is makred as (audience laughing) in certain utterance. Explain why audience laughed given the conversation, starting wtih 'The audience laughed because '"
#     utt_dic = {}
#     input_attr = sitcom_attr_sample["Video clips"]
#     for kk in input_attr.keys():
#         utt_dic[kk]=input_attr[kk]["Utterance"]
#
#     inputs = prompt + " video clip:" + str(utt_dic)
#     sitcom_reasoning.append({'id': f"{key}", "conversations": [{'from': "human", "value": inputs}, {'from': "gpt", "value": sitcom_gt_sample}]})
#
#
# # save new reasoning data
# with open("/local_data2/sung/dataset/ted_reasoning_train_ablation.json", "w") as f:
#     json.dump(sitcom_reasoning, f, indent=2)


# sitcom_reasoning = []
# for idx, key in enumerate(train_key):
#     sitcom_attr_sample = sitcom_attr_all[key]
#     sitcom_gt_sample = sitcom_gt[key]
#
#     prompt = f"Reasoning task: you are going to answer why the audience laughed given the video clip. The video clip from {sitcom_attr_sample['Video title']} is given with utterances for each clip."
#     question = "Give a detailed reason why the audience laugh, marked as (audience laughed), at most 30 words, starting with `the audience laughed because'"
#     utt_dic = {}
#     input_attr = sitcom_attr_sample["Video clips"]
#     for kk in input_attr.keys():
#         utt_dic[kk] = input_attr[kk]["Utterance"]
#
#     inputs = prompt + "Given video clip:" +str(utt_dic) + question
#     sitcom_reasoning.append({'id': f"{key}", "conversations": [{'from': "human", "value": inputs},
#                                                                {'from': "gpt", "value": sitcom_gt_sample}]})
#
# # save new reasoning data
# with open("/local_data2/sung/dataset/ted_reasoning_train_ablation_v2.json", "w") as f:
#     json.dump(sitcom_reasoning, f, indent=2)

