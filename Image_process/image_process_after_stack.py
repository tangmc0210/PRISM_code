import os
from pathlib import Path
from unittest.mock import patch
import pandas as pd
from tqdm import tqdm
import numpy as np
from lib.utils.io_utils import get_tif_list
from lib.fstack import stack_cyc
from lib.cidre import cidre_walk
from lib.register import register_meta
from lib.stitch import patch_tiles
from lib.stitch import template_stitch
from lib.stitch import stitch_offset
from lib.os_snippets import try_mkdir
from lib.register import register_manual
from lib.stitch import stitch_manual
from skimage.transform import resize
from skimage.util import img_as_uint

import shutil
from skimage.io import imread
from skimage.io import imsave


SRC_DIR = Path(r"F:\spatial_data\raw")
BASE_DIR = Path(r"F:\spatial_data\processed")
RUN_ID = '20240708_PRISM30_TNBC_BZ02_CA2_TCR+mut'
src_dir = SRC_DIR / RUN_ID
dest_dir = BASE_DIR / f'{RUN_ID}_processed'

aif_dir = dest_dir / 'focal_stacked'
sdc_dir = dest_dir / 'background_corrected'
rgs_dir = dest_dir / 'registered'
stc_dir = dest_dir / 'stitched'
rsz_dir = dest_dir / 'resized'

TileX, TileY = 13, 11


def resize_pad(img, size):
    img_resized = resize(img, size, anti_aliasing=True)
    img_padded = np.zeros(img.shape)
    y_start, x_start = (img.shape[0] - size[0]) // 2, (img.shape[1] - size[1]) // 2
    img_padded[y_start:y_start+size[0], x_start:x_start+size[1]] = img_resized
    img_padded = img_as_uint(img_padded)
    return img_padded


def resize_dir(in_dir, out_dir, chn):
    Path(out_dir).mkdir(exist_ok=True)
    chn_sizes = {'cy3': 2302, 'TxRed': 2303, 'FAM': 2301, 'DAPI': 2300}
    size = chn_sizes[chn]
    im_list = list(Path(in_dir).glob(f'*.tif'))
    for im_path in tqdm(im_list, desc=Path(in_dir).name):
        im = imread(im_path)
        im = resize_pad(im, (size, size))
        imsave(Path(out_dir)/im_path.name, im, check_contrast=False)


def resize_batch(in_dir, out_dir):
    try_mkdir(out_dir)
    cyc_paths = list(Path(in_dir).glob('cyc_*_*'))
    for cyc_path in cyc_paths:
        chn = cyc_path.name.split('_')[-1]
        if chn == 'cy5': shutil.copytree(cyc_path, Path(out_dir)/cyc_path.name)
        else: resize_dir(cyc_path, Path(out_dir)/cyc_path.name, chn)


def main():
    # raw_cyc_list = list(src_dir.glob('cyc_*'))
    # for cyc in raw_cyc_list:
    #   cyc_num = int(cyc.name.split('_')[1])
    #   stack_cyc(src_dir, aif_dir, cyc_num)

    cidre_walk(aif_dir, sdc_dir)

    rgs_dir.mkdir(exist_ok=True)
    ref_cyc = 1
    ref_chn = 'cy3'
    ref_chn_1 = 'cy5'
    ref_chn_2 = 'FAM'
    ref_chn_3 = 'TxRed'
    ref_dir = sdc_dir / f'cyc_{ref_cyc}_{ref_chn}'
    im_names = get_tif_list(ref_dir)

    meta_df = register_meta(str(sdc_dir), str(rgs_dir), ['cy3', 'cy5'], im_names, ref_cyc, ref_chn)
    meta_df.to_csv(rgs_dir / 'integer_offsets.csv')
    # register_manual(rgs_dir/'cyc_1_cy3', sdc_dir/'cyc_1_cy5', rgs_dir/'cyc_1_cy5') #
    register_manual(rgs_dir/'cyc_1_cy3', sdc_dir / 'cyc_1_FAM', rgs_dir/'cyc_1_FAM')
    register_manual(rgs_dir/'cyc_1_cy3', sdc_dir / 'cyc_1_TxRed', rgs_dir/'cyc_1_TxRed')
    register_manual(rgs_dir/'cyc_1_cy3', sdc_dir/'cyc_1_DAPI', rgs_dir/'cyc_1_DAPI')  # 0103 revised! Please remove this !
    
    patch_tiles(rgs_dir/f'cyc_{ref_cyc}_{ref_chn}', TileX * TileY)

    resize_batch(rgs_dir, rsz_dir)

    stc_dir.mkdir(exist_ok=True)
    template_stitch(rsz_dir/f'cyc_{ref_cyc}_{ref_chn_1}', stc_dir, TileX, TileY)

    offset_df = pd.read_csv(rgs_dir / 'integer_offsets.csv', index_col=0)
    # offset_df = offset_df.set_index('Unnamed: 0')
    # offset_df.index.name = None


    stitch_offset(rgs_dir, stc_dir, offset_df)

    # register_manual(rgs_dir/'cyc_1_cy3', sdc_dir/'cyc_1_FAM', rgs_dir/'cyc_1_FAM')
    # register_manual(rgs_dir/'cyc_1_cy3', sdc_dir/'cyc_1_TxRed', rgs_dir/'cyc_1_TxRed')
    # stitch_manual(rgs_dir/'cyc_1_FAM', stc_dir, offset_df, 10, bleed=30)
    # stitch_manual(rgs_dir/'cyc_1_TxRed', stc_dir, offset_df, 10, bleed=30)
    # im = imread(stc_dir/'cyc_11_DAPI.tif')
    # im_crop = im[10000:20000,10000:20000]
    # imsave(stc_dir/'cyc_11_DAPI_crop.tif', im_crop)


def test_rsz_before_rgs():
    cidre_walk(aif_dir, sdc_dir)
    resize_batch(sdc_dir, rsz_dir)

    rgs_dir.mkdir(exist_ok=True)
    ref_cyc = 1
    ref_chn_rgs = 'cy3' # cy5, FAM, TeRed
    ref_dir = rsz_dir / f'cyc_{ref_cyc}_{ref_chn_rgs}'
    im_names = get_tif_list(ref_dir)

    meta_df = register_meta(str(rsz_dir), str(rgs_dir), ['cy3', 'cy5'], im_names, ref_cyc, ref_chn_rgs)
    meta_df.to_csv(rgs_dir / 'integer_offsets.csv')
    register_manual(rgs_dir/f'cyc_1_{ref_chn_rgs}', rsz_dir/'cyc_1_FAM', rgs_dir/'cyc_1_FAM')
    register_manual(rgs_dir/f'cyc_1_{ref_chn_rgs}', rsz_dir/'cyc_1_TxRed', rgs_dir/'cyc_1_TxRed')
    register_manual(rgs_dir/f'cyc_1_{ref_chn_rgs}', rsz_dir/'cyc_1_DAPI', rgs_dir/'cyc_1_DAPI')  # 0103 revised! Please remove this !
    patch_tiles(rgs_dir/f'cyc_{ref_cyc}_{ref_chn_rgs}', TileX * TileY)

    ref_chn_stc = 'cy5'
    stc_dir.mkdir(exist_ok=True)
    template_stitch(rgs_dir/f'cyc_{ref_cyc}_{ref_chn_stc}', stc_dir, TileX, TileY)

    offset_df = pd.read_csv(rgs_dir / 'integer_offsets.csv', index_col=0)
    stitch_offset(rgs_dir, stc_dir, offset_df)


if __name__ == "__main__":
    # copy this file to the dest_dir
    current_file_path = os.path.abspath(__file__)
    target_file_path = os.path.join(dest_dir, os.path.basename(current_file_path))
    try: shutil.copy(current_file_path, target_file_path)
    except shutil.SameFileError: print('The file already exists in the destination directory.')
    except PermissionError: print("Permission denied: Unable to copy the file.")
    except FileNotFoundError: print("File not found: Source file does not exist.")
    except Exception as e: print(f"An error occurred: {e}")
    print('RUN_ID:', RUN_ID)
    test_rsz_before_rgs()
