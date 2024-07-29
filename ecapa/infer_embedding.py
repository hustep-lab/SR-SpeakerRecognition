from ECAPAModel import ECAPAModel
import torch
import soundfile as sf
import numpy as np
import torch.nn.functional as F
import os
import tqdm
import pickle
import pandas as pd
from tqdm import tqdm

def get_embedding(model, file_path):
    '''
    input: model with params
             path to .wav file
    output: normalized embedding of the audio
    '''

    audio, _ = sf.read(file_path)
    audio = torch.FloatTensor(np.stack([audio], axis=0)).cuda()

    with torch.no_grad():
        embedding = model.speaker_encoder.forward(audio, aug=False)
        embedding = F.normalize(embedding, p=2, dim=1)  # normalize

    return embedding.cpu().detach().numpy()

def gen_embedding_from_dir(model, dir, save_path):
    embeddings = {}
    for speaker in os.listdir(dir):
        for utt in os.listdir(os.path.join(dir, speaker)):
            try:
                embeddings[os.path.join(dir, speaker, utt)] = get_embedding(model, os.path.join(dir, speaker, utt))
            except:
                continue
            
    with open(save_path, 'wb') as handle:
        pickle.dump(embeddings, handle, protocol=pickle.HIGHEST_PROTOCOL)


model = ECAPAModel(lr=0.0005, lr_decay=.97, C=1024 , n_class=880, m=.2, s=30, test_step=1)
model = torch.compile(model)
model.load_parameters('PATH_TO_MODEL')
model.to('cuda')
model.eval()


gen_embedding_from_dir(model, 'DIRECTORY_TO_WAVS_CONTAINER', 'PATH_TO_BE_SAVED')
