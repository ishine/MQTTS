import json
import os
from random import shuffle
from tqdm import tqdm

metadata_path = "/Users/xingyijin/Downloads/WenetSpeech.json"


print ('Loading Labelfile...')
with open(str(metadata_path), 'r') as f:
    labels = json.load(f)


all_sid=[]
total_cnt = 0
skip_cnt = 0
for audiofile in tqdm(labels['audios']):
    for k, sentence in tqdm(enumerate(audiofile['segments'])):
        confidence = sentence['confidence']

        if confidence >=0.95:
            sentence_sid = sentence['sid']
            all_sid.append(sentence_sid)
            text = sentence['text']
            total_cnt += 1
            print("text count :", total_cnt, "skip count :", skip_cnt)
        else:
            skip_cnt += 1

shuffle(all_sid)

os.makedirs("datasets", exist_ok=True)
with open("datasets/training.txt", 'w') as f:
    for sid in all_sid[:-1000]:
        f.write(sid+'\n')

with open("datasets/validation.txt", 'w') as f:
    for sid in all_sid[-1000:]:
        f.write(sid+'\n')

