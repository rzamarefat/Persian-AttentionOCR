import torch
import torch.nn as nn

import random
from PIL import Image

from torchvision import transforms

from captcha.image import ImageCaptcha

from attention_ocr import OCR
from train_util import train_batch, eval_batch
from tokenizer import Tokenizer

img_width = 160
img_height = 60
max_len = 30

nh = 512

device = 'cpu'

# chars = list('1234567890abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ')
chars = list("لثمپچةسقۀژگأضوفذصن آظحبشیادءرؤغکئطخزعه_تج")
# gen = ImageCaptcha(img_width, img_height)
# n_chars = 4

tokenizer = Tokenizer(chars)
model = OCR(img_width, img_height, nh, tokenizer.n_token,
                max_len + 1, tokenizer.SOS_token, tokenizer.EOS_token).to(device=device)

model.load_state_dict(torch.load('/home/rmarefat/Desktop/git/attention-ocr/ckpts/time_2023-01-20_17-24-14_epoch_10.pth'))
img_trans = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=(0.229, 0.224, 0.225)),
])

# content = [random.randrange(0, len(chars)) for _ in range(n_chars)]
# s = ''.join([chars[i] for i in content])
# d = gen.generate(s)

d = Image.open("/home/rmarefat/Desktop/git/attention-ocr/00001.png")
d = d.convert("RGB")
d = d.resize((160, 60))
# d = Image.open(d)

model.eval()
with torch.no_grad():
    model_input = img_trans(d).unsqueeze(0)
    print(model_input.shape)
    pred = model(model_input)
    
rst = tokenizer.translate(pred.squeeze(0).argmax(1))
print(rst)