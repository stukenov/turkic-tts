import argparse
import json
import os
import re

import numpy as np
import torch
from attrdict import AttrDict
from num2words import num2words
from pydub import AudioSegment

from text import convert_text
from model import GradTTSWithEmo
from models import Generator as HiFiGAN
import utils_data


HIFIGAN_CONFIG = './configs/hifigan-config.json'
HIFIGAN_CHECKPT = './pre_trained_3/g_01720000'


if __name__ == '__main__':
    hps, args = utils_data.get_hparams_decode()
    device = torch.device('cpu')
    ckpt = args.model
    print(f"Loading checkpoint: {ckpt}")
    model = GradTTSWithEmo(**hps.model).to(device)
    utils_data.load_checkpoint(ckpt, model, None)
    model.eval()

    print('Initializing HiFi-GAN...')
    with open(HIFIGAN_CONFIG) as f:
        h = AttrDict(json.load(f))
    vocoder = HiFiGAN(h)
    # Fixed line to avoid KeyError: 'generator'
    state_dict = torch.load(HIFIGAN_CHECKPT, map_location='cpu')
    if 'generator' in state_dict:
        vocoder.load_state_dict(state_dict['generator'])
    else:
        vocoder.load_state_dict(state_dict)
    _ = vocoder.eval()
    vocoder.remove_weight_norm()

    emos = sorted(["angry", "surprise", "fear", "happy", "neutral", "sad"])
    speakers = ['M1', 'F1', 'M2']

    with open(args.file, 'r', encoding='utf-8') as f:
        texts = [line.strip() for line in f.readlines()]

    replace_nums = []
    for i in texts:
        replace_nums.append(i.split('|', 1))

    nums2word = [
        re.sub('(\d+)', lambda m: num2words(m.group(), lang='kz'), sentence)
        for sentence in np.array(replace_nums)[:, 0]
    ]
    # Speakers id.
    # M1 = 0
    # F1 = 1
    # M2 = 2
    text2speech = []
    for i, j in zip(nums2word, np.array(replace_nums)[:, 1]):
        text2speech.append(f'{i}|{j}')

    for i, line in enumerate(text2speech):
        emo_i = int(line.split('|')[1])
        control_spk_id = int(line.split('|')[2])
        control_emo_id = emos.index(emos[emo_i])
        text = line.split('|')[0]
        with torch.no_grad():
            # define emotion
            emo = torch.LongTensor([control_emo_id]).to(device)
            sid = torch.LongTensor([control_spk_id]).to(device)
            text_padded, text_len = convert_text(text)
            y_enc, y_dec, attn = model.forward(
                text_padded,
                text_len,
                n_timesteps=args.timesteps,
                temperature=args.noise,
                stoc=args.stoc,
                spk=sid,
                emo=emo,
                length_scale=1.,
                classifier_free_guidance=args.guidance
            )
        res = y_dec.squeeze().numpy()
        x = torch.from_numpy(res).unsqueeze(0)
        y_g_hat = vocoder(x)
        audio = y_g_hat.squeeze()
        audio = audio * 32768.0
        audio = audio.detach().numpy().astype('int16')
        audio = AudioSegment(
            audio.data,
            frame_rate=22050,
            sample_width=2,
            channels=1
        )
        audio.export(
            f'{args.generated_path}/output.wav',
            format="wav"
        )
