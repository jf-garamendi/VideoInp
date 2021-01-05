import torch
from tqdm import  tqdm
import glob
import os

class dataset(torch.utils.data.Dataset):
    def __init__(self, config, val=False, test=False):
        super(dataset, self).__init__()
        self.n = config.n_frames
        self.size = config.IMAGE_SHAPE

        self.flow_filenames = glob.glob(os.path.join(config.flow_dir, "*.flo"))
        self.flow_filenames.sort()

        self.






    def __len__(self):

        return len(self.data_items)

    def __getitem__(self, idx):
        flow_dir, img_dir, mask_dir, m, M, rm, rM = self.data_items[idx]

        flow_set = []
        rflow_set = []
        mask_set = []
        edge_set = []
        flow_masked_set = []
        rflow_masked_set = []
        img_set = []

        for t, i in enumerate(range(-self.n // 2 + 6, 6 + self.n // 2)):
            tmp_flow = cv2.imread(flow_dir[i])[:, :, :2] / 255 * (M[i] - m[i]) + m[i]
            tmp_flow = self._flow_tf(tmp_flow)

            if self.config.rflow:  # rflows are sometimes considered as flows since the sequence in reverse time is also valable. so we have to use actual flows as rflows for such sequences
                if flow_dir[i][-5] != 'r':
                    tmp_rflow = cv2.imread(flow_dir[i][:-4] + 'r.png')[:, :, :2] / 255 * (rM[i] - rm[i]) + rm[i]

                else:
                    tmp_rflow = cv2.imread(flow_dir[i][:-5] + '.png')[:, :, :2] / 255 * (M[i] - m[i]) + m[i]
                tmp_rflow = self._flow_tf(tmp_rflow)

            if self.config.edge == 'grad':
                img = cv2.imread(boundary_dir[i])
                img = cv2.resize(img, (self.size[1], self.size[0]), interpolation=cv2.INTER_NEAREST) / 127.5 - 1

                tmp_boundary = img[:, :, 0:1]

            elif self.config.edge == 'hed':
                img = cv2.imread(boundary_dir[i])
                img = cv2.resize(img, (self.size[1], self.size[0]), interpolation=cv2.INTER_NEAREST) / 127.5 - 1
                tmp_boundary = img[:, :, 0:1]  # .reshape(self.size[0],self.size[1],1)

            elif self.config.edge == 'canny':
                img = cvb.flow2rgb(flow=tmp_flow)
                tmp_boundary = np.expand_dims(canny(img.mean(-1), sigma=.5), -1) * 2 - 1
                # plt.figure()
                # plt.imshow(tmp_boundary[:,:,0])
                # plt.show()

            elif self.config.edge == 'none':
                # print(boundary_dir[i])
                img = cv2.imread(boundary_dir[i])
                img = cv2.resize(img, (self.size[1], self.size[0])) / 127.5 - 1
                tmp_boundary = img[:, :, 0:1]  # .reshape(self.size[0],self.size[1],1)*0.

            if self.config.image:
                path = img_dir[i]
                if path[-3:] == 'png':
                    path = path[:-3] + 'jpg'
                img = cv2.imread(path)

                tmp_img = self._img_tf(img)

            if t == 0:
                if self.config.MASK_MODE == 'bbox':
                    if self.config.bias_edge:
                        img_shape = self.config.IMAGE_SHAPE
                        img_height = img_shape[0]
                        img_width = img_shape[1]

                        max_h = img_height - self.config.VERTICAL_MARGIN - self.config.MASK_HEIGHT // 2
                        max_w = img_width - self.config.HORIZONTAL_MARGIN - self.config.MASK_WIDTH // 2

                        grad_norm = np.sqrt(np.sum((tmp_flow[1 + self.config.MASK_HEIGHT // 2:max_h,
                                                    self.config.MASK_WIDTH // 2:max_w - 1] - tmp_flow[
                                                                                             self.config.MASK_HEIGHT // 2:max_h - 1,
                                                                                             self.config.MASK_WIDTH // 2:max_w - 1]) ** 2 + (
                                                               tmp_flow[self.config.MASK_HEIGHT // 2:max_h - 1,
                                                               1 + self.config.MASK_WIDTH // 2:max_w] - tmp_flow[
                                                                                                        self.config.MASK_HEIGHT // 2:max_h - 1,
                                                                                                        self.config.MASK_WIDTH // 2:max_w - 1]) ** 2,
                                                   axis=2))
                        xx, yy = np.meshgrid(np.arange(grad_norm.shape[0]), np.arange(grad_norm.shape[1]))
                        ind = list(np.stack((xx, yy), axis=-1).reshape(-1, 2))

                        t, l = ind[sorted(np.arange(len(ind)), key=lambda k: grad_norm[int(ind[k][0]), int(ind[k][1])],
                                          reverse=True)[min(len(ind) - 1, int(np.random.exponential(0.1 * len(ind))))]]

                        tmp_bbox = (t, l, self.config.MASK_WIDTH, self.config.MASK_WIDTH)

                    else:
                        tmp_bbox = im.random_bbox(self.config)
                    tmp_mask = im.bbox2mask_perso(self.config, tmp_bbox)
                    tmp_mask = tmp_mask[0, 0, :, :]
                    fix_mask = np.expand_dims(tmp_mask, axis=2)
                elif self.config.MASK_MODE == 'mid-bbox':
                    tmp_mask = im.mid_bbox_mask(self.config)
                    tmp_mask = tmp_mask[0, 0, :, :]
                    fix_mask = np.expand_dims(tmp_mask, axis=2)

            if self.config.get_mask:
                tmp_mask = cv2.imread(mask_dir[i],
                                      cv2.imread_UNCHANGED)
                tmp_mask = self._mask_tf(tmp_mask)
            elif self.config.FIX_MASK:
                tmp_mask = fix_mask.copy()
            else:

                tmp_mask = im.bbox2mask_perso(self.config, tmp_bbox)

                tmp_mask = tmp_mask[0, 0, :, :]
                tmp_mask = np.expand_dims(tmp_mask, axis=2)

            if self.config.sandwich and (i == -self.n // 2 + 6 or i == 6 + self.n // 2 - 1):
                tmp_mask = tmp_mask * 0.

            if self.config.two_masks and t != 2 and t != 3:
                tmp_mask = tmp_mask * 0.

            tmp_flow_masked = tmp_flow * (1. - tmp_mask)
            if self.config.rflow:
                tmp_rflow_masked = tmp_rflow * (1. - tmp_mask)

            if self.config.INITIAL_HOLE:
                img_dir = self.data_items[idx][3]
                tmp_flow_resized = cv2.resize(tmp_flow, (self.size[1] // 2, self.size[0] // 2))

                tmp_mask_resized = cv2.resize(tmp_mask, (self.size[1] // 2, self.size[0] // 2), cv2.INTER_NEAREST)

                tmp_flow_masked_small = tmp_flow_resized
                tmp_flow_masked_small[:, :, 0] = rf.regionfill(tmp_flow_resized[:, :, 0], tmp_mask_resized)
                tmp_flow_masked_small[:, :, 1] = rf.regionfill(tmp_flow_resized[:, :, 1], tmp_mask_resized)
                tmp_flow_masked = tmp_flow_masked + \
                                  tmp_mask * cv2.resize(tmp_flow_masked_small, (self.size[1], self.size[0]))

                if self.config.rflow:
                    tmp_rflow_resized = cv2.resize(tmp_rflow, (self.size[1] // 2, self.size[0] // 2))
                    tmp_rflow_masked_small = tmp_rflow_resized
                    tmp_rflow_masked_small[:, :, 0] = rf.regionfill(tmp_rflow_resized[:, :, 0], tmp_mask_resized)
                    tmp_rflow_masked_small[:, :, 1] = rf.regionfill(tmp_rflow_resized[:, :, 1], tmp_mask_resized)
                    tmp_rflow_masked = tmp_rflow_masked + \
                                       tmp_mask * cv2.resize(tmp_rflow_masked_small, (self.size[1], self.size[0]))

            if self.config.bbox:
                flow_masked_set.append(
                    tmp_flow_masked[tmp_bbox[0]:tmp_bbox[0] + tmp_bbox[2], tmp_bbox[1]:tmp_bbox[1] + tmp_bbox[3]])
                flow_set.append(tmp_flow[tmp_bbox[0]:tmp_bbox[0] + tmp_bbox[2], tmp_bbox[1]:tmp_bbox[1] + tmp_bbox[3]])
                mask_set.append(np.concatenate((tmp_mask[tmp_bbox[0]:tmp_bbox[0] + tmp_bbox[2],
                                                tmp_bbox[1]:tmp_bbox[1] + tmp_bbox[3]],
                                                tmp_mask[tmp_bbox[0]:tmp_bbox[0] + tmp_bbox[2],
                                                tmp_bbox[1]:tmp_bbox[1] + tmp_bbox[3]]), axis=2))

                if self.config.edge is not None:
                    edge_set.append(
                        tmp_boundary[tmp_bbox[0]:tmp_bbox[0] + tmp_bbox[2], tmp_bbox[1]:tmp_bbox[1] + tmp_bbox[3]])
                if self.config.rflow:
                    rflow_masked_set.append(
                        tmp_rflow_masked[tmp_bbox[0]:tmp_bbox[0] + tmp_bbox[2], tmp_bbox[1]:tmp_bbox[1] + tmp_bbox[3]])
                    rflow_set.append(
                        tmp_rflow[tmp_bbox[0]:tmp_bbox[0] + tmp_bbox[2], tmp_bbox[1]:tmp_bbox[1] + tmp_bbox[3]])
                if self.config.image:
                    img_set.append(
                        tmp_img[tmp_bbox[0]:tmp_bbox[0] + tmp_bbox[2], tmp_bbox[1]:tmp_bbox[1] + tmp_bbox[3]])

            else:
                flow_masked_set.append(tmp_flow_masked)
                flow_set.append(tmp_flow)
                mask_set.append(np.concatenate((tmp_mask, tmp_mask), axis=2))

                if self.config.edge is not None:
                    edge_set.append(tmp_boundary)
                if self.config.rflow:
                    rflow_masked_set.append(tmp_rflow_masked)
                    rflow_set.append(tmp_rflow)
                if self.config.image:
                    img_set.append(tmp_img)

        flow_masked = np.stack(flow_masked_set, axis=2)
        gt_flow = np.stack(flow_set, axis=2)
        mask = np.stack(mask_set, axis=2)

        flow_masked = torch.from_numpy(flow_masked).permute(2, 3, 0, 1).contiguous().float()
        gt_flow = torch.from_numpy(gt_flow).permute(2, 3, 0, 1).contiguous().float()
        mask = torch.from_numpy(mask).permute(2, 3, 0, 1).contiguous().float()

        if self.config.rflow:
            rflow_masked = torch.from_numpy(np.stack(rflow_masked_set, axis=2)).permute(2, 3, 0, 1).contiguous().float()
            gt_rflow = torch.from_numpy(np.stack(rflow_set, axis=2)).permute(2, 3, 0, 1).contiguous().float()
        else:
            rflow_masked = 0
            gt_rflow = 0

        if self.config.image:
            img = torch.from_numpy(np.stack(img_set, axis=2)).permute(2, 3, 0, 1).contiguous().float()
        else:
            img = 0

        if self.config.edge is not None:
            edge = torch.from_numpy(np.stack(edge_set, axis=2)).permute(2, 3, 0, 1).contiguous().float()
        else:
            edge = 0

        return flow_masked, rflow_masked, mask, gt_flow, gt_rflow, edge, img, tmp_bbox

    def _img_tf(self, img):
        img = cv2.resize(img, (self.size[1], self.size[0]))
        img = img / 127.5 - 1

        return img

    def _mask_tf(self, mask):
        mask = cv2.resize(mask, (self.size[1], self.size[0]),
                          interpolation=cv2.INTER_NEAREST)
        if self.config.enlarge_mask:
            enlarge_kernel = np.ones((self.config.enlarge_kernel, self.config.enlarge_kernel),
                                     np.uint8)
            tmp_mask = cv2.dilate(mask[:, :, 0], enlarge_kernel, iterations=1)
            mask[(tmp_mask > 0), :] = 255

        mask = mask[:, :, 0]
        mask = np.expand_dims(mask, axis=2)
        mask = mask / 255
        return mask

    def _flow_tf(self, flow):
        origin_shape = flow.shape
        flow = cv2.resize(flow, (self.size[1], self.size[0]))
        flow[:, :, 0] = flow[:, :, 0].clip(-1. * origin_shape[1], origin_shape[1]) / origin_shape[1] * self.size[1]
        flow[:, :, 1] = flow[:, :, 1].clip(-1. * origin_shape[0], origin_shape[0]) / origin_shape[0] * self.size[0]

        return flow


