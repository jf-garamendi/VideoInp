import numpy as np
import cv2
from utils.utils_from_FGVC.frame_inpaint import DeepFillv1
from utils.utils_from_FGVC.get_flowNN_gradient import get_flowNN_gradient
from argparse import Namespace
import scipy.ndimage
from utils.utils_from_FGVC.Poisson_blend_img import Poisson_blend_img
from utils.utils_from_FGVC.spatial_inpaint import spatial_inpaint

def gradient_mask(mask):

    gradient_mask = np.logical_or.reduce((mask,
        np.concatenate((mask[1:, :], np.zeros((1, mask.shape[1]), dtype=np.bool)), axis=0),
        np.concatenate((mask[:, 1:], np.zeros((mask.shape[0], 1), dtype=np.bool)), axis=1)))

    return gradient_mask

def from_flow_to_frame(frames, flows, masks):
    #inputs are tensors

    nFrame, C, imgH, imgW = flows.shape

    video = frames.clone().detach().cpu().permute(2,3,1,0).numpy()
    masks = np.squeeze(masks.clone().detach().cpu().permute(1,2,3,0).numpy().astype(int).astype(bool))
    masks_dilated = masks

    # translate flow into Forward flow and backward flow
    flows = flows.clone().detach().cpu().permute(2, 3, 1, 0).numpy()
    videoFlowF =  flows[:,:,0:2,0:-1]
    videoFlowB = flows[:,:,2:,1:]

    # Prepare gradients
    gradient_x = np.empty(((imgH, imgW, 3, 0)), dtype=np.float32)
    gradient_y = np.empty(((imgH, imgW, 3, 0)), dtype=np.float32)

    for indFrame in range(nFrame):
        print("from_flow_to_frame: ", indFrame)
        img = video[:, :, :, indFrame]
        img[masks[:, :, indFrame], :] = 0
        img = cv2.inpaint((img * 255).astype(np.uint8), masks[:, :, indFrame].astype(np.uint8), 3, cv2.INPAINT_TELEA).astype(np.float32)  / 255.

        gradient_x_ = np.concatenate((np.diff(img, axis=1), np.zeros((imgH, 1, 3), dtype=np.float32)), axis=1)
        gradient_y_ = np.concatenate((np.diff(img, axis=0), np.zeros((1, imgW, 3), dtype=np.float32)), axis=0)
        gradient_x = np.concatenate((gradient_x, gradient_x_.reshape(imgH, imgW, 3, 1)), axis=-1)
        gradient_y = np.concatenate((gradient_y, gradient_y_.reshape(imgH, imgW, 3, 1)), axis=-1)

        gradient_x[masks_dilated[:, :, indFrame], :, indFrame] = 0
        gradient_y[masks_dilated[:, :, indFrame], :, indFrame] = 0

    iter = 0
    mask_tofill = masks
    gradient_x_filled = gradient_x  # corrupted gradient_x, mask_gradient indicates the missing gradient region
    gradient_y_filled = gradient_y  # corrupted gradient_y, mask_gradient indicates the missing gradient region
    mask_gradient = masks_dilated
    video_comp = video

    # Image inpainting model.
    # TODO: Remove hardcoded pretrained_model path
    deepfill = DeepFillv1(pretrained_model="../weight/imagenet_deepfill.pth", image_shape=[imgH, imgW])

    while (np.sum(masks) > 0):
        #default args from FGVC code:
        args = Namespace(Nonlocal=False, consistencyThres=float('inf'), alpha=0.1)
        # Gradient propagation.
        gradient_x_filled, gradient_y_filled, mask_gradient = \
            get_flowNN_gradient(args,
                                gradient_x_filled,
                                gradient_y_filled,
                                masks,
                                mask_gradient,
                                videoFlowF,
                                videoFlowB,
                                None,
                                None)

        # if there exist holes in mask, Poisson blending will fail. So I did this trick. I sacrifice some value. Another solution is to modify Poisson blending.
        for indFrame in range(nFrame):
            mask_gradient[:, :, indFrame] = scipy.ndimage.binary_fill_holes(mask_gradient[:, :, indFrame]).astype(
                np.bool)

        # After one gradient propagation iteration
        # gradient --> RGB
        for indFrame in range(nFrame):
            print("Poisson blending frame {0:3d}".format(indFrame))

            if masks[:, :, indFrame].sum() > 0:
                try:
                    frameBlend, UnfilledMask = Poisson_blend_img(video_comp[:, :, :, indFrame],
                                                                 gradient_x_filled[:, 0: imgW - 1, :, indFrame],
                                                                 gradient_y_filled[0: imgH - 1, :, :, indFrame],
                                                                 masks[:, :, indFrame], mask_gradient[:, :, indFrame])
                except:
                    frameBlend, UnfilledMask = video_comp[:, :, :, indFrame], masks[:, :, indFrame]

                frameBlend = np.clip(frameBlend, 0, 1.0)
                tmp = cv2.inpaint((frameBlend * 255).astype(np.uint8), UnfilledMask.astype(np.uint8), 3,
                                  cv2.INPAINT_TELEA).astype(np.float32) / 255.
                frameBlend[UnfilledMask, :] = tmp[UnfilledMask, :]

                video_comp[:, :, :, indFrame] = frameBlend
                masks[:, :, indFrame] = UnfilledMask


        masks, video_comp = spatial_inpaint(deepfill, masks, video_comp)
        iter += 1

        # Re-calculate gradient_x/y_filled and mask_gradient
        for indFrame in range(nFrame):
            mask_gradient[:, :, indFrame] = gradient_mask(masks[:, :, indFrame])

            gradient_x_filled[:, :, :, indFrame] = np.concatenate(
                (np.diff(video_comp[:, :, :, indFrame], axis=1), np.zeros((imgH, 1, 3), dtype=np.float32)), axis=1)
            gradient_y_filled[:, :, :, indFrame] = np.concatenate(
                (np.diff(video_comp[:, :, :, indFrame], axis=0), np.zeros((1, imgW, 3), dtype=np.float32)), axis=0)

            gradient_x_filled[mask_gradient[:, :, indFrame], :, indFrame] = 0
            gradient_y_filled[mask_gradient[:, :, indFrame], :, indFrame] = 0

    return video_comp