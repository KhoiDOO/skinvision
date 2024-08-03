from PIL import Image
from io import BytesIO
from tqdm import tqdm

import os
import h5py
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import imageio.v3 as iio
import math


SAFE_PIX_RAG = 0
M2P = 3.7795275591


if __name__ == '__main__':
    data_dir = '/media/mountHDD3/data_storage/biomedical_data/isic/2024'
    trimg_dir = os.path.join(data_dir, 'train-image')
    trhdf5 = os.path.join(data_dir, 'train-image.hdf5')
    tshdf5 = os.path.join(data_dir, 'test-image.hdf5')
    trmeta = os.path.join(data_dir, 'train-metadata.csv')
    tsmeta = os.path.join(data_dir, 'test-metadata.csv')
    subform = os.path.join(data_dir, 'sample_submission.csv')

    sv_dir = os.path.join(data_dir, f'img_crop_{SAFE_PIX_RAG}')
    os.makedirs(sv_dir, exist_ok=True)

    trdf = pd.read_csv(trmeta)
    trhp = h5py.File(trhdf5, 'r')

    ids = trdf['isic_id'].values.tolist()
    for id in tqdm(ids):
        img = np.array(Image.open(BytesIO(trhp[id][()])))
        H, W, C = img.shape
        d = trdf[trdf['isic_id'] == id]['clin_size_long_diam_mm'].values.item()
        crop_range = math.ceil(d * M2P) + SAFE_PIX_RAG
        if crop_range > H:
            print(f'crop_range: {crop_range} bigger than size: {H} with id: {id}')
        crop_img = img[H//2 - crop_range:H//2+crop_range, W//2 - crop_range:W//2+crop_range, :]

        iio.imwrite(f'{sv_dir}/{id}.png', crop_img)