import json
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from tqdm import tqdm

metadata_path = "out.json"

print ('Loading Labelfile...')
with open(str(metadata_path), 'r') as f:
    labels = json.load(f)


inference_pipeline = pipeline(
    task=Tasks.punctuation,
    model='damo/punc_ct-transformer_zh-cn-common-vocab272727-pytorch'
)


total_cnt = 0
skip_cnt = 0
for audiofile in tqdm(labels['audios']):
    for k, sentence in tqdm(enumerate(audiofile['segments'])):
        confidence = sentence['confidence']
        if confidence >=0.95:
            sentence_sid = sentence['sid']
            text = sentence['text']
            rec_result = inference_pipeline(text_in=text)
            print(rec_result)
            sentence['text'] = rec_result['text']
            total_cnt += 1
            print("preprocessed text count :", total_cnt, "skip count :", skip_cnt)
        else:
            skip_cnt += 1

with open("WenetSpeech_output.json", 'w') as f:
    json.dump(labels, f, ensure_ascii=False, indent=2)



