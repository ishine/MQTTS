import json

import librosa
from tqdm import tqdm

from text.cleaner import clean_text
train_transcription_path = "../../Downloads/vits_filelist_genshin_xm37/vits_filelist_genshin_xm37_new.txt"
val_transcription_path = "../../Downloads/vits_filelist_genshin_xm37/vits_filelist_genshin_xm37_new.txt"
duration = 0

labels = {}
with open("datasets/training.txt", 'w') as f1:
    for line in tqdm(open(train_transcription_path).readlines()):
        path, sid,text = line.strip().split('|')
        name = path.split('/')[-1].split('.')[0]
        f1.write(name+'\n')
        phones = clean_text(text)
        phones = " ".join(phones)
        # duration = librosa.get_duration(filename=f"datasets/audios/{name}.wav")
        labels[f"{name}.wav"] = {
            "text": text,
            "phoneme": phones,
            "sid": sid,
            "duration": duration
        }
with open("datasets/train.json", 'w') as f2:
    json.dump(labels, f2, indent=2, ensure_ascii=False)



labels = {}
with open("datasets/validation.txt", 'w') as f1:
    for line in tqdm(open(val_transcription_path).readlines()):
        path, sid,text = line.strip().split('|')
        name = path.split('/')[-1].split('.')[0]
        f1.write(name+'\n')
        phones = clean_text(text)
        phones = " ".join(phones)
        # duration = librosa.get_duration(filename=f"datasets/audios/{name}.wav")
        labels[f"{name}.wav"] = {
            "text": text,
            "phoneme": phones,
            "sid": sid,
            "duration": duration
        }
with open("datasets/dev.json", 'w') as f2:
    json.dump(labels, f2, indent=2, ensure_ascii=False)
