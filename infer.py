from utils import *
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import argparse

parser = argparse.ArgumentParser(description='My training script.')
parser.add_argument('--finetuned', type=bool, default=True)
args = parser.parse_args()

tokenizer = AutoTokenizer.from_pretrained("/root/ChatYuan-large-v2")
model_path = "./outputs/model_files/" if args.finetuned else "/root/ChatYuan-large-v2"
model_tuned = AutoModelForSeq2SeqLM.from_pretrained(model_path)

"""
for name, para in model_tuned.named_parameters():
    print(name, para.shape)
"""

"""
def weights_init_normal(m):
    '''Takes in a module and initializes all linear layers with weight
       values taken from a normal distribution.'''

    classname = m.__class__.__name__
    # for every Linear layer in a model
    if classname.find('Linear') != -1:
        y = m.in_features
    # m.weight.data shoud be taken from a normal distribution
        m.weight.data.normal_(0.0,1/np.sqrt(y))
    # m.bias.data should be 0
model_tuned.apply(weights_init_normal)
"""

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # cuda
model_tuned.to(device)

def preprocess(text):
    return text.replace("\n", "_")
def postprocess(text):
    return text.replace("_", "\n")

def answer(model, text, sample=True, top_p=0.9, temperature=0.7, context = ""):
    text = f"{context}\n用户：{text}\n小元："
    text = text.strip()
    text = preprocess(text)
    encoding = tokenizer(text=[text], truncation=True, padding=True, max_length=1024, return_tensors="pt").to(device) 
    if not sample:
        out = model.generate(**encoding, return_dict_in_generate=True, output_scores=False, max_new_tokens=1024, num_beams=1, length_penalty=0.6)
    else:
        out = model.generate(**encoding, return_dict_in_generate=True, output_scores=False, max_new_tokens=1024, do_sample=True, top_p=top_p, temperature=temperature, no_repeat_ngram_size=12)
    out_text = tokenizer.batch_decode(out["sequences"], skip_special_tokens=True)
    return postprocess(out_text[0])


if __name__ == '__main__':
    history = []
    while True:
        query = input("\n用户：")
        if len(query) == 0:
            break
        context = "\n".join(history[-5:])
        response = answer(model_tuned, query, context=context)
        history.append(f"用户：{query}\n小元：{response}")
        print(f"小元：{response}")