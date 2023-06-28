from __future__ import print_function, absolute_import
import numpy as np
import os
import os.path as osp
import cv2

__all__ = ['visualize_ranked_results']

GRID_SPACING = 10
QUERY_EXTRA_SPACING = 90
BW = 5 # border width
GREEN = (0, 255, 0)
RED = (0, 0, 255)


def visualize_ranked_results(distmat, width=128, height=256, save_dir='data/places365_large/viz', topk=10):
    """Visualizes ranked results.

    Supports both image-reid

    For image-reid, ranks will be plotted in a single figure. 

    Args:
        distmat (numpy.ndarray): distance matrix of shape (num_query, num_gallery).
        dataset (tuple): a 2-tuple containing (query, gallery), each of which contains
            tuples of (img_path(s), pid, camid, dsetid).
        data_type (str): "image" or "video".
        width (int, optional): resized image width. Default is 128.
        height (int, optional): resized image height. Default is 256.
        save_dir (str): directory to save output images.
        topk (int, optional): denoting top-k images in the rank list to be visualized.
            Default is 10.
    """
    num_q, num_g = distmat.shape
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    print('# query: {}\n# gallery {}'.format(num_q, num_g))
    print('Visualizing top-{} ranks ...'.format(topk))

    ##query, gallery = dataset
    ##assert num_q == len(query)
    ##assert num_g == len(gallery)

    indices = np.argsort(distmat, axis=1)
    print(indices)

    """def _cp_img_to(src, dst, rank, prefix, matched=False):
        
        Args:
            src: image path
            dst: target directory
            rank: int, denoting ranked position, starting from 1
            prefix: string
            matched: bool
        
        if isinstance(src, (tuple, list)):
            if prefix == 'gallery':
                suffix = 'TRUE' if matched else 'FALSE'
                dst = osp.join(
                    dst, prefix + '_top' + str(rank).zfill(3)
                ) + '_' + suffix
            else:
                dst = osp.join(dst, prefix + '_top' + str(rank).zfill(3))
            mkdir_if_missing(dst)
            for img_path in src:
                shutil.copy(img_path, dst)
        else:
            dst = osp.join(
                dst, prefix + '_top' + str(rank).zfill(3) + '_name_' +
                osp.basename(src)
            )
            shutil.copy(src, dst)
    """
    for q_idx in range(num_q):

        ##Additional code added here:##
        qpid = int(q_idx//400)
        qimgid_within_class = (q_idx % 400) + 1
        qimg_path = os.path.join("data/places365_large/test/gallery/000"+str(qpid),"0000"+str(4500+qimgid_within_class)+".jpg")
        ####

        qimg = cv2.imread(qimg_path)
        qimg = cv2.resize(qimg, (width, height))
        qimg = cv2.copyMakeBorder(qimg, BW, BW, BW, BW, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        # resize twice to ensure that the border width is consistent across images
        qimg = cv2.resize(qimg, (width, height))
        num_cols = topk + 1
        grid_img = 255 * np.ones(
            (height, num_cols*width + topk*GRID_SPACING + QUERY_EXTRA_SPACING, 3),
            dtype=np.uint8
        )
        grid_img[:, :width, :] = qimg

        rank_idx = 1
        for g_idx in indices[q_idx, 1:]: #1: to skip first image which is definitely itself
            gpid = int(g_idx//400)
            gimgid_within_class = (g_idx % 400) + 1
            gimg_path = os.path.join("data/places365_large/test/gallery/000"+str(gpid),"0000"+str(4500+gimgid_within_class)+".jpg")

            matched = gpid == qpid
            
            border_color = GREEN if matched else RED
            gimg = cv2.imread(gimg_path)
            gimg = cv2.resize(gimg, (width, height))
            gimg = cv2.copyMakeBorder(
                gimg,
                BW,
                BW,
                BW,
                BW,
                cv2.BORDER_CONSTANT,
                value=border_color
            )
            gimg = cv2.resize(gimg, (width, height))
            start = rank_idx*width + rank_idx*GRID_SPACING + QUERY_EXTRA_SPACING
            end = (
                rank_idx+1
            ) * width + rank_idx*GRID_SPACING + QUERY_EXTRA_SPACING
            grid_img[:, start:end, :] = gimg

            rank_idx += 1
            if rank_idx > topk:
                break



        classID = qimg_path.split('/')[-2]
        imname = osp.basename(osp.splitext(qimg_path)[0])
        cv2.imwrite(osp.join(save_dir, 'c' + classID + '_' + imname + '.jpg'), grid_img)

        if (q_idx+1) % 100 == 0:
            print('- done {}/{}'.format(q_idx + 1, num_q))

    print('Done. Images have been saved to "{}" ...'.format(save_dir))
