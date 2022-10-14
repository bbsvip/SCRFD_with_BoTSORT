""" Created by MrBBS """
# 10/13/2022
# -*-encoding:utf-8-*-

import onnxruntime
import os.path as osp
import cv2
import numpy as np
from numpy.linalg import inv, norm, lstsq
from numpy.linalg import matrix_rank as rank
from tracker.bot_sort import BoTSORT
import random


class FaceAlign:
    def __init__(self, output_size=(112, 112), desiredLeftEye=(0.5, 0.40)):
        self.output_size = output_size
        self.desiredLeftEye = desiredLeftEye
        self.ref_pts = np.array([[-1.58083929e-01, -3.84258929e-02],
                                 [1.56533929e-01, -4.01660714e-02],
                                 [2.25000000e-04, 1.40505357e-01],
                                 [-1.29024107e-01, 3.24691964e-01],
                                 [1.31516964e-01, 3.23250893e-01]])

    def tformfwd(self, trans, uv):
        """
        Function:
        ----------
            apply affine transform 'trans' to uv
        Parameters:
        ----------
            @trans: 3x3 np.array
                transform matrix
            @uv: Kx2 np.array
                each row is a pair of coordinates (x, y)
        Returns:
        ----------
            @xy: Kx2 np.array
                each row is a pair of transformed coordinates (x, y)
        """
        uv = np.hstack((
            uv, np.ones((uv.shape[0], 1))
        ))
        xy = np.dot(uv, trans)
        xy = xy[:, 0:-1]
        return xy

    def findNonreflectiveSimilarity(self, uv, xy, options=None):
        """
        Function:
        ----------
            Find Non-reflective Similarity Transform Matrix 'trans':
                u = uv[:, 0]
                v = uv[:, 1]
                x = xy[:, 0]
                y = xy[:, 1]
                [x, y, 1] = [u, v, 1] * trans
        Parameters:
        ----------
            @uv: Kx2 np.array
                source points each row is a pair of coordinates (x, y)
            @xy: Kx2 np.array
                each row is a pair of inverse-transformed
            @option: not used, keep it as None
        Returns:
            @trans: 3x3 np.array
                transform matrix from uv to xy
            @trans_inv: 3x3 np.array
                inverse of trans, transform matrix from xy to uv
        Matlab:
        ----------
        % For a nonreflective similarity:
        %
        % let sc = s*cos(theta)
        % let ss = s*sin(theta)
        %
        %                   [ sc -ss
        % [u v] = [x y 1] *   ss  sc
        %                     tx  ty]
        %
        % There are 4 unknowns: sc,ss,tx,ty.
        %
        % Another way to write this is:
        %
        % u = [x y 1 0] * [sc
        %                  ss
        %                  tx
        %                  ty]
        %
        % v = [y -x 0 1] * [sc
        %                   ss
        %                   tx
        %                   ty]
        %
        % With 2 or more correspondence points we can combine the u equations and
        % the v equations for one linear system to solve for sc,ss,tx,ty.
        %
        % [ u1  ] = [ x1  y1  1  0 ] * [sc]
        % [ u2  ]   [ x2  y2  1  0 ]   [ss]
        % [ ... ]   [ ...          ]   [tx]
        % [ un  ]   [ xn  yn  1  0 ]   [ty]
        % [ v1  ]   [ y1 -x1  0  1 ]
        % [ v2  ]   [ y2 -x2  0  1 ]
        % [ ... ]   [ ...          ]
        % [ vn  ]   [ yn -xn  0  1 ]
        %
        % Or rewriting the above matrix equation:
        % U = X * r, where r = [sc ss tx ty]'
        % so r = X\ U.
        %
        """
        options = {'K': 2}

        K = options['K']
        M = xy.shape[0]
        x = xy[:, 0].reshape((-1, 1))  # use reshape to keep a column vector
        y = xy[:, 1].reshape((-1, 1))  # use reshape to keep a column vector
        # print '--->x, y:\n', x, y

        tmp1 = np.hstack((x, y, np.ones((M, 1)), np.zeros((M, 1))))
        tmp2 = np.hstack((y, -x, np.zeros((M, 1)), np.ones((M, 1))))
        X = np.vstack((tmp1, tmp2))
        # print '--->X.shape: ', X.shape
        # print 'X:\n', X

        u = uv[:, 0].reshape((-1, 1))  # use reshape to keep a column vector
        v = uv[:, 1].reshape((-1, 1))  # use reshape to keep a column vector
        U = np.vstack((u, v))
        # print '--->U.shape: ', U.shape
        # print 'U:\n', U

        # We know that X * r = U
        if rank(X) >= 2 * K:
            r, _, _, _ = lstsq(X, U, rcond=None)
            r = np.squeeze(r)
        else:
            raise Exception('cp2tform:twoUniquePointsReq')

        # print '--->r:\n', r

        sc = r[0]
        ss = r[1]
        tx = r[2]
        ty = r[3]

        Tinv = np.array([
            [sc, -ss, 0],
            [ss, sc, 0],
            [tx, ty, 1]
        ])

        # print '--->Tinv:\n', Tinv

        T = inv(Tinv)
        # print '--->T:\n', T

        T[:, 2] = np.array([0, 0, 1])

        return T, Tinv

    def findSimilarity(self, uv, xy, options=None):
        """
        Function:
        ----------
            Find Reflective Similarity Transform Matrix 'trans':
                u = uv[:, 0]
                v = uv[:, 1]
                x = xy[:, 0]
                y = xy[:, 1]
                [x, y, 1] = [u, v, 1] * trans
        Parameters:
        ----------
            @uv: Kx2 np.array
                source points each row is a pair of coordinates (x, y)
            @xy: Kx2 np.array
                each row is a pair of inverse-transformed
            @option: not used, keep it as None
        Returns:
        ----------
            @trans: 3x3 np.array
                transform matrix from uv to xy
            @trans_inv: 3x3 np.array
                inverse of trans, transform matrix from xy to uv
        Matlab:
        ----------
        % The similarities are a superset of the nonreflective similarities as they may
        % also include reflection.
        %
        % let sc = s*cos(theta)
        % let ss = s*sin(theta)
        %
        %                   [ sc -ss
        % [u v] = [x y 1] *   ss  sc
        %                     tx  ty]
        %
        %          OR
        %
        %                   [ sc  ss
        % [u v] = [x y 1] *   ss -sc
        %                     tx  ty]
        %
        % Algorithm:
        % 1) Solve for trans1, a nonreflective similarity.
        % 2) Reflect the xy data across the Y-axis,
        %    and solve for trans2r, also a nonreflective similarity.
        % 3) Transform trans2r to trans2, undoing the reflection done in step 2.
        % 4) Use TFORMFWD to transform uv using both trans1 and trans2,
        %    and compare the results, Returnsing the transformation corresponding
        %    to the smaller L2 norm.
        % Need to reset options.K to prepare for calls to findNonreflectiveSimilarity.
        % This is safe because we already checked that there are enough point pairs.
        """
        options = {'K': 2}

        #    uv = np.array(uv)
        #    xy = np.array(xy)

        # Solve for trans1
        trans1, trans1_inv = self.findNonreflectiveSimilarity(uv, xy, options)

        # Solve for trans2

        # manually reflect the xy data across the Y-axis
        xyR = xy
        xyR[:, 0] = -1 * xyR[:, 0]

        trans2r, trans2r_inv = self.findNonreflectiveSimilarity(uv, xyR, options)

        # manually reflect the tform to undo the reflection done on xyR
        TreflectY = np.array([
            [-1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ])

        trans2 = np.dot(trans2r, TreflectY)

        # Figure out if trans1 or trans2 is better
        xy1 = self.tformfwd(trans1, uv)
        norm1 = norm(xy1 - xy)

        xy2 = self.tformfwd(trans2, uv)
        norm2 = norm(xy2 - xy)

        if norm1 <= norm2:
            return trans1, trans1_inv
        else:
            trans2_inv = inv(trans2)
            return trans2, trans2_inv

    def get_similarity_transform(self, src_pts, dst_pts, reflective=True):
        """
        Function:
        ----------
            Find Similarity Transform Matrix 'trans':
                u = src_pts[:, 0]
                v = src_pts[:, 1]
                x = dst_pts[:, 0]
                y = dst_pts[:, 1]
                [x, y, 1] = [u, v, 1] * trans
        Parameters:
        ----------
            @src_pts: Kx2 np.array
                source points, each row is a pair of coordinates (x, y)
            @dst_pts: Kx2 np.array
                destination points, each row is a pair of transformed
                coordinates (x, y)
            @reflective: True or False
                if True:
                    use reflective similarity transform
                else:
                    use non-reflective similarity transform
        Returns:
        ----------
           @trans: 3x3 np.array
                transform matrix from uv to xy
            trans_inv: 3x3 np.array
                inverse of trans, transform matrix from xy to uv
        """

        if reflective:
            trans, trans_inv = self.findSimilarity(src_pts, dst_pts)
        else:
            trans, trans_inv = self.findNonreflectiveSimilarity(src_pts, dst_pts)

        return trans, trans_inv

    def cvt_tform_mat_for_cv2(self, trans):
        """
        Function:
        ----------
            Convert Transform Matrix 'trans' into 'cv2_trans' which could be
            directly used by cv2.warpAffine():
                u = src_pts[:, 0]
                v = src_pts[:, 1]
                x = dst_pts[:, 0]
                y = dst_pts[:, 1]
                [x, y].T = cv_trans * [u, v, 1].T
        Parameters:
        ----------
            @trans: 3x3 np.array
                transform matrix from uv to xy
        Returns:
        ----------
            @cv2_trans: 2x3 np.array
                transform matrix from src_pts to dst_pts, could be directly used
                for cv2.warpAffine()
        """
        cv2_trans = trans[:, 0:2].T

        return cv2_trans

    def get_similarity_transform_for_cv2(self, src_pts, dst_pts, reflective=True):
        """
        Function:
        ----------
            Find Similarity Transform Matrix 'cv2_trans' which could be
            directly used by cv2.warpAffine():
                u = src_pts[:, 0]
                v = src_pts[:, 1]
                x = dst_pts[:, 0]
                y = dst_pts[:, 1]
                [x, y].T = cv_trans * [u, v, 1].T
        Parameters:
        ----------
            @src_pts: Kx2 np.array
                source points, each row is a pair of coordinates (x, y)
            @dst_pts: Kx2 np.array
                destination points, each row is a pair of transformed
                coordinates (x, y)
            reflective: True or False
                if True:
                    use reflective similarity transform
                else:
                    use non-reflective similarity transform
        Returns:
        ----------
            @cv2_trans: 2x3 np.array
                transform matrix from src_pts to dst_pts, could be directly used
                for cv2.warpAffine()
        """
        trans, trans_inv = self.get_similarity_transform(src_pts, dst_pts, reflective)
        cv2_trans = self.cvt_tform_mat_for_cv2(trans)

        return cv2_trans

    def align(self, img, src_pts, scale=1.0, transpose_input=False):
        w, h = self.output_size

        # Actual offset = new center - old center (scaled)
        scale_ = max(w, h) * scale
        cx_ref = cy_ref = 0.
        offset_x = 0.5 * w - cx_ref * scale_
        offset_y = 0.5 * h - cy_ref * scale_

        s = np.array(src_pts).astype(np.float32).reshape([-1, 2])
        r = np.array(self.ref_pts).astype(np.float32) * scale_ + np.array([[offset_x, offset_y]])
        if transpose_input:
            s = s.reshape([2, -1]).T

        tfm = self.get_similarity_transform_for_cv2(s, r)
        dst_img = cv2.warpAffine(img, tfm, self.output_size)

        s_new = np.concatenate([s.reshape([2, -1]), np.ones((1, s.shape[0]))])
        s_new = np.matmul(tfm, s_new)
        s_new = s_new.reshape([-1]) if transpose_input else s_new.T.reshape([-1])
        tfm = tfm.reshape([-1])
        return dst_img, s_new, tfm


class FaceDetection:
    def __init__(self, model_file=None, cuda=False, thresh=0.5, nms_thresh=0.4):
        import onnxruntime
        self.model_file = model_file
        self.thresh = thresh
        self.batched = False
        self.cuda = cuda
        assert osp.exists(self.model_file)
        providers = ['CPUExecutionProvider']
        if self.cuda:
            providers = ['CUDAExecutionProvider']
        self.session = onnxruntime.InferenceSession(self.model_file, providers=providers)
        self.center_cache = {}
        self.nms_thresh = nms_thresh
        self._init_vars()

    def _init_vars(self):
        input_cfg = self.session.get_inputs()[0]
        input_shape = input_cfg.shape
        if isinstance(input_shape[2], str):
            self.input_size = None
        else:
            self.input_size = tuple(input_shape[2:4][::-1])
        input_name = input_cfg.name
        outputs = self.session.get_outputs()
        if len(outputs[0].shape) == 3:
            self.batched = True
        output_names = []
        for o in outputs:
            output_names.append(o.name)
        self.input_name = input_name
        self.output_names = output_names
        self.fmc = 3
        self._feat_stride_fpn = [8, 16, 32]
        self._num_anchors = 2
        self.use_kps = True

    @staticmethod
    def distance2bbox(points, distance, max_shape=None):
        """Decode distance prediction to bounding box.
        Args:
            points (Tensor): Shape (n, 2), [x, y].
            distance (Tensor): Distance from the given point to 4
                boundaries (left, top, right, bottom).
            max_shape (tuple): Shape of the image.
        Returns:
            Tensor: Decoded bboxes.
        """
        x1 = points[:, 0] - distance[:, 0]
        y1 = points[:, 1] - distance[:, 1]
        x2 = points[:, 0] + distance[:, 2]
        y2 = points[:, 1] + distance[:, 3]
        if max_shape is not None:
            x1 = x1.clamp(min=0, max=max_shape[1])
            y1 = y1.clamp(min=0, max=max_shape[0])
            x2 = x2.clamp(min=0, max=max_shape[1])
            y2 = y2.clamp(min=0, max=max_shape[0])
        return np.stack([x1, y1, x2, y2], axis=-1)

    @staticmethod
    def distance2kps(points, distance, max_shape=None):
        """Decode distance prediction to bounding box.
        Args:
            points (Tensor): Shape (n, 2), [x, y].
            distance (Tensor): Distance from the given point to 4
                boundaries (left, top, right, bottom).
            max_shape (tuple): Shape of the image.
        Returns:
            Tensor: Decoded bboxes.
        """
        preds = []
        for i in range(0, distance.shape[1], 2):
            px = points[:, i % 2] + distance[:, i]
            py = points[:, i % 2 + 1] + distance[:, i + 1]
            if max_shape is not None:
                px = px.clamp(min=0, max=max_shape[1])
                py = py.clamp(min=0, max=max_shape[0])
            preds.append(px)
            preds.append(py)
        return np.stack(preds, axis=-1)

    def forward(self, img):
        scores_list = []
        bboxes_list = []
        kpss_list = []
        input_size = tuple(img.shape[0:2][::-1])
        blob = cv2.dnn.blobFromImage(img, 1.0 / 128, input_size, (127.5, 127.5, 127.5), swapRB=True)
        net_outs = self.session.run(self.output_names, {self.input_name: blob})

        input_height = blob.shape[2]
        input_width = blob.shape[3]
        fmc = self.fmc
        for idx, stride in enumerate(self._feat_stride_fpn):
            if self.batched:
                scores = net_outs[idx][0]
                bbox_preds = net_outs[idx + fmc][0]
                bbox_preds = bbox_preds * stride
                kps_preds = net_outs[idx + fmc * 2][0] * stride
            else:
                scores = net_outs[idx]
                bbox_preds = net_outs[idx + fmc]
                bbox_preds = bbox_preds * stride
                kps_preds = net_outs[idx + fmc * 2] * stride

            height = input_height // stride
            width = input_width // stride
            key = (height, width, stride)
            if key in self.center_cache:
                anchor_centers = self.center_cache[key]
            else:
                anchor_centers = np.stack(np.mgrid[:height, :width][::-1], axis=-1).astype(np.float32)
                anchor_centers = (anchor_centers * stride).reshape((-1, 2))
                if self._num_anchors > 1:
                    anchor_centers = np.stack([anchor_centers] * self._num_anchors, axis=1).reshape((-1, 2))
                if len(self.center_cache) < 100:
                    self.center_cache[key] = anchor_centers

            pos_inds = np.where(scores >= self.thresh)[0]
            bboxes = self.distance2bbox(anchor_centers, bbox_preds)
            pos_scores = scores[pos_inds]
            pos_bboxes = bboxes[pos_inds]
            scores_list.append(pos_scores)
            bboxes_list.append(pos_bboxes)
            kpss = self.distance2kps(anchor_centers, kps_preds)
            kpss = kpss.reshape((kpss.shape[0], -1, 2))
            pos_kpss = kpss[pos_inds]
            kpss_list.append(pos_kpss)
        return scores_list, bboxes_list, kpss_list

    def detect(self, img, input_size=(640, 640), max_num=0, metric='default'):
        assert input_size is not None or self.input_size is not None
        input_size = self.input_size if input_size is None else input_size

        im_ratio = float(img.shape[0]) / img.shape[1]
        model_ratio = float(input_size[1]) / input_size[0]
        if im_ratio > model_ratio:
            new_height = input_size[1]
            new_width = int(new_height / im_ratio)
        else:
            new_width = input_size[0]
            new_height = int(new_width * im_ratio)
        det_scale = float(new_height) / img.shape[0]
        resized_img = cv2.resize(img, (new_width, new_height))
        det_img = np.zeros((input_size[1], input_size[0], 3), dtype=np.uint8)
        det_img[:new_height, :new_width, :] = resized_img

        scores_list, bboxes_list, kpss_list = self.forward(det_img)

        scores = np.vstack(scores_list)
        scores_ravel = scores.ravel()
        order = scores_ravel.argsort()[::-1]
        bboxes = np.vstack(bboxes_list) / det_scale
        kpss = np.vstack(kpss_list) / det_scale
        pre_det = np.hstack((bboxes, scores)).astype(np.float32, copy=False)
        pre_det = pre_det[order, :]
        keep = self.nms(pre_det)
        det = pre_det[keep, :]
        kpss = kpss[order, :, :]
        kpss = kpss[keep, :, :]
        if 0 < max_num < det.shape[0]:
            area = (det[:, 2] - det[:, 0]) * (det[:, 3] -
                                              det[:, 1])
            img_center = img.shape[0] // 2, img.shape[1] // 2
            offsets = np.vstack([
                (det[:, 0] + det[:, 2]) / 2 - img_center[1],
                (det[:, 1] + det[:, 3]) / 2 - img_center[0]
            ])
            offset_dist_squared = np.sum(np.power(offsets, 2.0), 0)
            if metric == 'max':
                values = area
            else:
                values = area - offset_dist_squared * 2.0  # some extra weight on the centering
            bindex = np.argsort(
                values)[::-1]  # some extra weight on the centering
            bindex = bindex[0:max_num]
            det = det[bindex, :]
            if kpss is not None:
                kpss = kpss[bindex, :]
        return det, kpss

    def nms(self, dets):
        thresh = self.nms_thresh
        x1 = dets[:, 0]
        y1 = dets[:, 1]
        x2 = dets[:, 2]
        y2 = dets[:, 3]
        scores = dets[:, 4]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)

            inds = np.where(ovr <= thresh)[0]
            order = order[inds + 1]

        return keep


if __name__ == '__main__':
    import time


    def get_color(idx):
        idx = idx * 3
        color = ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)

        return color


    tracker = BoTSORT()
    detector = FaceDetection(model_file='onnx/scrfd_2.5g_bnkps.onnx', thresh=0.5)
    face_align = FaceAlign()
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, img = cap.read()
        crop_img = img.copy()
        st = time.time()
        bboxes, kpss = detector.detect(img, input_size=(512, 512))
        try:
            results_tracking = tracker.update(bboxes, kpss, crop_img)
            cv2.putText(img, str(round(1 / (time.time() - st))), (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            for t in results_tracking:
                color = get_color(t.track_id)
                x, y, w, h = t.tlwh
                x1, y1, x2, y2 = tuple(map(int, (x, y, x + w, y + h)))
                tid = t.track_id
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                cv2.putText(img, str(t.score), (x2, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                for i, kp in enumerate(t.kp):
                    kp = kp.astype(np.int)
                    if i == 2:
                        cv2.putText(img, str(tid), kp, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                    cv2.circle(img, tuple(kp), 1, color, 2)
        except:
            pass
        cv2.imshow('cc', img)
        if cv2.waitKey(1) & 0xff == ord('q'):
            break
