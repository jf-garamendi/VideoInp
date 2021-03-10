import glob
import os
import torch
import numpy as np
from PIL import Image
from os.path import *
import re
from utils.flow_viz import flow_to_image

import cv2
import io

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

TAG_CHAR = np.array([202021.25], np.float32)

def create_dir(dir):
    """Creates a directory if not exist.
    """
    if not os.path.exists(dir):
        os.makedirs(dir)

def load_mask(path_to_mask):
    pil_mask = Image.open(path_to_mask).convert('L')

    mask = np.array(pil_mask).astype(np.uint8)
    mask = (mask > 0).astype(np.uint8)

    return mask

def load_video_frames_as_tensor(video_path):
    # Loads frames.
    frame_filename_list = glob.glob(os.path.join(video_path, '*.png')) + \
                          glob.glob(os.path.join(video_path, '*.jpg'))

    video = []
    for filename in sorted(frame_filename_list):
        # TODO: check permute dimensions
        video.append(torch.from_numpy(np.array(Image.open(filename)).astype(np.uint8)).permute(2, 0, 1).float())

    video = torch.stack(video, dim=0)

    return video

def tensor_save_flow_and_img(flow, folder):
    #TODO: Check Dimensions
    #flow: nd_array of size (N,C,H,W)
    folder_flow = join(folder, 'flow_flo')
    folder_img = join(folder, 'flow_png')

    create_dir(folder_flow)
    create_dir(folder_img)

    for i in range(flow.shape[0]):
        flow_frame = torch.squeeze(flow[i,:,:,:]).permute((1,2,0))
        flow_frame = flow_frame.detach().numpy()

        flow_img = flow_to_image(flow_frame)
        flow_img = Image.fromarray(flow_img)

        # Saves the flow and flow_img.
        writeFlow(os.path.join(folder_flow, '%05d.flo' % i), flow_frame)
        flow_img.save(os.path.join(folder_img, '%05d.png' % i))

def save_flow_and_img(flow, folder_flow='flow_flo', folder_img='flow_png'):
    #TODO: Check Dimensions
    #flow: list of matrices (each matrix is the flow of a frame) of size (H,W,C)

    create_dir(folder_flow)
    create_dir(folder_img)

    for i in range(len(flow)):
        flow_frame = flow[i]

        flow_img = flow_to_image(flow_frame)
        flow_img = Image.fromarray(flow_img)

        # Saves the flow and flow_img.
        writeFlow(os.path.join(folder_flow, '%05d.flo' % i), flow_frame)
        flow_img.save(os.path.join(folder_img, '%05d.png' % i))


def read_flow(path):
    #adapted from
    # Author : George Gach (@georgegach)
    # Date   : July 2019

    # Adapted from the Middlebury Vision project's Flow-Code
    # URL    : http://vision.middlebury.edu/flow/

    TAG_FLOAT = 202021.25

    if not isinstance(path, io.BufferedReader):
        if not isinstance(path, str):
            raise AssertionError(
                "Input [{p}] is not a string".format(p=path))
        if not os.path.isfile(path):
            raise AssertionError(
                "Path [{p}] does not exist".format(p=path))
        if not path.split('.')[-1] == 'flo':
            raise AssertionError(
                "File extension [flo] required, [{f}] given".format(f=path.split('.')[-1]))

        flo = open(path, 'rb')
    else:
        flo = path

    tag = np.frombuffer(flo.read(4), np.float32, count=1)[0]
    if not TAG_FLOAT == tag:
        raise AssertionError("Wrong Tag [{t}]".format(t=tag))

    width = np.frombuffer(flo.read(4), np.int32, count=1)[0]
    if not (width > 0 and width < 100000):
        raise AssertionError("Illegal width [{w}]".format(w=width))

    height = np.frombuffer(flo.read(4), np.int32, count=1)[0]
    if not (width > 0 and width < 100000):
        raise AssertionError("Illegal height [{h}]".format(h=height))

    nbands = 2
    tmp = np.frombuffer(flo.read(nbands * width * height * 4),
                        np.float32, count=nbands * width * height)
    flow = np.resize(tmp, (int(height), int(width), int(nbands)))
    flo.close()

    return flow


def readPFM(file):
    file = open(file, 'rb')

    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().rstrip()
    if header == b'PF':
        color = True
    elif header == b'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(rb'^(\d+)\s(\d+)\s$', file.readline())
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().rstrip())
    if scale < 0:  # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>'  # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    return data


def writeFlow(filename, uv, v=None):
    """ Write optical flow to file.

    If v is None, uv is assumed to contain both u and v channels,
    stacked in depth.
    Original code by Deqing Sun, adapted from Daniel Scharstein.
    """
    nBands = 2

    if v is None:
        assert (uv.ndim == 3)
        assert (uv.shape[2] == 2)
        u = uv[:, :, 0]
        v = uv[:, :, 1]
    else:
        u = uv

    assert (u.shape == v.shape)
    height, width = u.shape
    f = open(filename, 'wb')
    # write the header
    f.write(TAG_CHAR)
    np.array(width).astype(np.int32).tofile(f)
    np.array(height).astype(np.int32).tofile(f)
    # arrange into matrix form
    tmp = np.zeros((height, width * nBands))
    tmp[:, np.arange(width) * 2] = u
    tmp[:, np.arange(width) * 2 + 1] = v
    tmp.astype(np.float32).tofile(f)
    f.close()


def readFlowKITTI(filename):
    flow = cv2.imread(filename, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)
    flow = flow[:, :, ::-1].astype(np.float32)
    flow, valid = flow[:, :, :2], flow[:, :, 2]
    flow = (flow - 2 ** 15) / 64.0
    return flow, valid


def readDispKITTI(filename):
    disp = cv2.imread(filename, cv2.IMREAD_ANYDEPTH) / 256.0
    valid = disp > 0.0
    flow = np.stack([-disp, np.zeros_like(disp)], -1)
    return flow, valid


def writeFlowKITTI(filename, uv):
    uv = 64.0 * uv + 2 ** 15
    valid = np.ones([uv.shape[0], uv.shape[1], 1])
    uv = np.concatenate([uv, valid], axis=-1).astype(np.uint16)
    cv2.imwrite(filename, uv[..., ::-1])


def read_gen(file_name, pil=False):
    ext = splitext(file_name)[-1]
    if ext == '.png' or ext == '.jpeg' or ext == '.ppm' or ext == '.jpg':
        return Image.open(file_name)
    elif ext == '.bin' or ext == '.raw':
        return np.load(file_name)
    elif ext == '.flo':
        return readFlow(file_name).astype(np.float32)
    elif ext == '.pfm':
        flow = readPFM(file_name).astype(np.float32)
        if len(flow.shape) == 2:
            return flow
        else:
            return flow[:, :, :-1]
    return []