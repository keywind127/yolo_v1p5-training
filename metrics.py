# YOLO Metrics
from tensorflow.python.framework.ops import EagerTensor
from tensorflow.keras.backend import epsilon
import tensorflow as tf 
from typing import *
import numpy 

class MetricUtils:
    @staticmethod
    def tensor_intersection_over_union(bounding_box_1 : EagerTensor, bounding_box_2 : EagerTensor) -> EagerTensor:
        """
            Parameters:
                [ 1 ] bounding_box_1 : (N, S, S, 4)
                [ 2 ] bounding_box_2 : (N, S, S, 4)
                
            Limitations:
                [ 1 ] all values (x, y, w, h) in both tensors must be within Range[0, 1]
        """

        # safeguards to ensure bounding boxes are bounded within Range[0, 1]
        tf.debugging.assert_greater_equal(bounding_box_1, 0.0)
        tf.debugging.assert_greater_equal(bounding_box_2, 0.0)
        tf.debugging.assert_less_equal(bounding_box_1, 1.0)
        tf.debugging.assert_less_equal(bounding_box_2, 1.0)

        # box_area_1, box_area_2 : (N, S, S, 1) [ area of bounding box 1 and 2 ]
        box_area_1 = bounding_box_1[..., 2:3] * bounding_box_1[..., 3:4]
        box_area_2 = bounding_box_2[..., 2:3] * bounding_box_2[..., 3:4]

        # bounding_box_1, bounding_box_2 : (N, S, S, 4) [ bounding boxes converted from xywh to xyxy ]
        (bounding_box_1, bounding_box_2) = (
            tf.concat([
                bounding_box_1[..., 0:1] - bounding_box_1[..., 2:3] / 2,
                bounding_box_1[..., 1:2] - bounding_box_1[..., 3:4] / 2,
                bounding_box_1[..., 0:1] + bounding_box_1[..., 2:3] / 2,
                bounding_box_1[..., 1:2] + bounding_box_1[..., 3:4] / 2
            ], axis = 3),
            tf.concat([
                bounding_box_2[..., 0:1] - bounding_box_2[..., 2:3] / 2,
                bounding_box_2[..., 1:2] - bounding_box_2[..., 3:4] / 2,
                bounding_box_2[..., 0:1] + bounding_box_2[..., 2:3] / 2,
                bounding_box_2[..., 1:2] + bounding_box_2[..., 3:4] / 2
            ], axis = 3)
        )

        # x_min, x_max, y_min, y_max : (N, S, S, 1) [ find bounding box of intersection ]
        (x_min, x_max, y_min, y_max) = (
            tf.maximum(bounding_box_1[..., 0:1], bounding_box_2[..., 0:1]),
            tf.minimum(bounding_box_1[..., 2:3], bounding_box_2[..., 2:3]),
            tf.maximum(bounding_box_1[..., 1:2], bounding_box_2[..., 1:2]),
            tf.minimum(bounding_box_1[..., 3:4], bounding_box_2[..., 3:4])
        )

        # intersection : (N, S, S, 1) [ find intersection area ]
        intersection = tf.maximum(x_max - x_min, 0) * tf.maximum(y_max - y_min, 0)

        # union : (N, S, S, 1) [ find union area ]
        union = box_area_1 + box_area_2 - intersection 

        # intersection_over_union : (N, S, S, 1) [ find iou ]
        intersection_over_union = intersection / (union + epsilon())

        return intersection_over_union

    @staticmethod 
    def list_intersection_over_union(bounding_box_1 : List[ float ], bounding_box_2 : List[ float ]) -> float:
        """
            Limitations:
                [ 1 ] bounding_box_1 and bounding_box_2 must contain 4 values
        """

        assert len(bounding_box_1) == 4
        assert len(bounding_box_2) == 4

        (box_area_1, box_area_2) = (
            bounding_box_1[2] * bounding_box_1[3],
            bounding_box_2[2] * bounding_box_2[3]
        )

        (bounding_box_1, bounding_box_2) = ([ 
                bounding_box_1[0] - bounding_box_1[2] / 2,
                bounding_box_1[1] - bounding_box_1[3] / 2,
                bounding_box_1[0] + bounding_box_1[2] / 2,
                bounding_box_1[1] + bounding_box_1[3] / 2
            ], [ 
                bounding_box_2[0] - bounding_box_2[2] / 2,
                bounding_box_2[1] - bounding_box_2[3] / 2,
                bounding_box_2[0] + bounding_box_2[2] / 2,
                bounding_box_2[1] + bounding_box_2[3] / 2
            ]
        )

        (x_min, x_max, y_min, y_max) = (
            max(bounding_box_1[0], bounding_box_2[0]), 
            min(bounding_box_1[2], bounding_box_2[2]),
            max(bounding_box_1[1], bounding_box_2[1]), 
            min(bounding_box_1[3], bounding_box_2[3])
        )

        intersection = max(x_max - x_min, 0) * max(y_max - y_min, 0)

        return intersection / (box_area_1 + box_area_2 - intersection + epsilon())

    @classmethod 
    def reduce_bounding_boxes(class_, true_boxes : EagerTensor, pred_boxes : EagerTensor) -> EagerTensor:
        """
            Parameters:
                [ 1 ] true_boxes : (N, S, S, 5)
                [ 2 ] pred_boxes : (N, S, S, 2, 5)
            Returns:
                [ 1 ] best_boxes : (N, S, S, 1)
                [ 2 ] iou_scores : (N, S, S, 5)
        """

        # clipped_pred_boxes : (N, S, S, 2, 5) [ clip off values (x, y, w, h, c) out of Range[0, 1] ]
        clipped_pred_boxes = tf.clip_by_value(pred_boxes, 0, 1)

        # iou_scores : (N, S, S, 2) [ intersection over union for each bounding box with ground truth ]
        iou_scores = tf.concat([
            class_.tensor_intersection_over_union(true_boxes[..., 1:5], clipped_pred_boxes[..., 0, 1:5]),
            class_.tensor_intersection_over_union(true_boxes[..., 1:5], clipped_pred_boxes[..., 1, 1:5])
        ], axis = 3)

        # best_boxes : (N, S, S, 1) [ find the indices for bounding boxes with relatively higher iou scores ]
        best_boxes = tf.cast(tf.expand_dims(tf.argmax(iou_scores, axis = 3), axis = 3), tf.int32)

        # iou_scores : (N, S, S, 1) [ keep only the iou scores of the best bounding boxes ]
        iou_scores = tf.expand_dims(tf.reduce_max(iou_scores, axis = 3), axis = 3)

        # I, J, K : (N, S, S) [ respectively (img_idx, row_idx, col_idx) ]
        (I, J, K) = tf.meshgrid(tf.range(tf.shape(best_boxes)[0]), tf.range(tf.shape(best_boxes)[1]), tf.range(tf.shape(best_boxes)[2]), indexing = "ij")

        # I, J, K : (N, S, S, 1) [ expand the number of dimensions, and cast the indices as 32-bit integers ]
        (I, J, K) = (
            tf.cast(tf.expand_dims(I, axis = 3), tf.int32),
            tf.cast(tf.expand_dims(J, axis = 3), tf.int32),
            tf.cast(tf.expand_dims(K, axis = 3), tf.int32)
        )

        # best_boxes : (N, S, S, 4) [ calculate the complete indices for the best bounding boxes ]
        best_boxes = tf.concat([ I, J, K, best_boxes ], axis = 3)

        # best_boxes : (N, S, S, 5) [ extract only the best bounding boxes ]
        best_boxes = tf.gather_nd(pred_boxes, best_boxes)

        return (best_boxes, iou_scores)

    @staticmethod 
    def extract_bounding_boxes(label_data : EagerTensor, S : int, C : int, is_prediction : bool = True) -> EagerTensor:
        """
            Parameters:
                [ 1 ] label_data : (N, S, S, C + 10) or (N, S, S, C + 5) 
                [ 2 ] S { grid cell size }
                [ 3 ] is_prediction { whether if is prediction or label }
            Returns:
                [ 1 ] bounding_boxes : (N * S * S, 7) { (img_idx, obj, cls_idx, x, y, w, h) }
        """

        label_data = tf.cast(label_data, tf.float32)

        # label_data : (N, S, S, C + 10) or (N, S, S, C + 5) [ clip off values (x, y, w, h, c) out of Range[0, 1] ]
        label_data = tf.clip_by_value(label_data, 0, 1)

        # C : (N, S, S, 1) [ determine class for each grid cell ]
        T = tf.cast(tf.expand_dims(tf.argmax(label_data[..., 0 : C], axis = 3), axis = 3), tf.float32)

        # label_data : (N, S, S, 10) or (N, S, S, 5) [ remove class probabilities ]
        label_data = label_data[..., C : ]

        if (is_prediction):

            # label_data : (N, S, S, 2, 5) [ reshape the bounding boxes ]
            label_data = tf.reshape(label_data, (-1, S, S, 2, 5))

            # obj_indices : (N, S, S, 1) [ find the indices for bounding boxes with relatively higher objectness scores ]
            obj_indices = tf.cast(tf.expand_dims(tf.argmax(tf.concat([ 
                    label_data[..., 0, 0:1], label_data[..., 1, 0:1] 
                ], axis = 3), axis = 3), axis = 3), tf.int32)
            
            # I, J, K : (N, S, S) [ respectively (img_idx, row_idx, col_idx) ]
            (I, J, K) = tf.meshgrid(tf.range(tf.shape(obj_indices)[0]), tf.range(S), tf.range(S), indexing = "ij")

            # I, J, K : (N, S, S, 1) [ expand the number of dimensions, and cast the indices as 32-bit integers ]
            (I, J, K) = (
                tf.cast(tf.expand_dims(I, axis = 3), tf.int32),
                tf.cast(tf.expand_dims(J, axis = 3), tf.int32),
                tf.cast(tf.expand_dims(K, axis = 3), tf.int32)
            )

            # obj_indices : (N, S, S, 4) [ calculate the complete indices for the best bounding boxes ]
            obj_indices = tf.concat([ I, J, K, obj_indices ], axis = 3)

            # label_data : (N, S, S, 5) [ extract only the best bounding boxes ]
            label_data = tf.gather_nd(label_data, obj_indices)

        # I, J, K : (N, S, S) [ respectively (img_idx, row_idx, col_idx) ]
        (I, J, K) = tf.meshgrid(tf.range(tf.shape(label_data)[0]), tf.range(S), tf.range(S), indexing = "ij")

        # I, J, K : (N, S, S, 1) [ expand the number of dimensions, and cast the indices as 32-bit floats ]
        (I, J, K) = (
            tf.cast(tf.expand_dims(I, axis = 3), tf.float32),
            tf.cast(tf.expand_dims(J, axis = 3), tf.float32),
            tf.cast(tf.expand_dims(K, axis = 3), tf.float32)
        )

        # X, Y, W, H, O : (N, S, S, 1) [ calculate x-y relative to the entire image and extract w-h-o sub-tensors ]
        (X, Y, W, H, O) = (
            (label_data[..., 1:2] + K) / S,
            (label_data[..., 2:3] + J) / S,
            (label_data[..., 3:4]    )    ,
            (label_data[..., 4:5]    )    ,
            (label_data[..., 0:1]    )
        )

        # bounding_boxes : (N * S * S, 7) [ assemble bounding box info (img_idx, obj, cls_idx, x, y, w, h) and reshape the tensor ] 
        bounding_boxes = tf.reshape(tf.concat([  I, O, T, X, Y, W, H  ], axis = 3), (-1, 7))

        return bounding_boxes 

    @classmethod 
    def non_max_suppression(class_, bounding_boxes : List[ List[ float ] ], thresh_obj : float, thresh_iou : float):
        """
            Parameters:
                [ 1 ] bounding_boxes : (N * S * S, 7) { (img_idx, obj, cls_idx, x, y, w, h) }
            Limitations:
                [ 1 ] bounding_boxes must be of type list
        """

        assert isinstance(bounding_boxes, list)

        bounding_boxes = sorted(filter(lambda x : x[1] >= thresh_obj, bounding_boxes), key = lambda x : x[1], reverse = True)

        suppressed_bounding_boxes = []

        while (bounding_boxes):

            bounding_box = bounding_boxes.pop(0)

            bounding_boxes = [
                _bounding_box for _bounding_box in bounding_boxes
                    if ((bounding_box[0] != _bounding_box[0]) or (bounding_box[2] != _bounding_box[2]) or
                        (class_.list_intersection_over_union(bounding_box[-4:], _bounding_box[-4:]) < thresh_iou))
            ]

            suppressed_bounding_boxes.append(bounding_box) 

        return suppressed_bounding_boxes

    @classmethod 
    def extract_and_format_bounding_boxes(class_, pred_boxes : EagerTensor, S : int, C : int, thresh_obj : float = 0.5, thresh_iou : float = 0.5) -> numpy.ndarray:
        """
            Parameters:
                [ 1 ] pred_boxes : (N, S, S, C + 10)
                [ 2 ] S
                [ 3 ] C
                [ 4 ] thresh_obj 
                [ 5 ] thresh_iou
        """

        # extracted_bounding_boxes : (N * S * S, 7)
        extracted_bounding_boxes = class_.extract_bounding_boxes(pred_boxes, S, C) 

        # convert to list of list [ [ img_idx, obj_scr, cls_idx, x, y, w, h ] ]
        extracted_bounding_boxes = extracted_bounding_boxes.numpy().tolist()

        # suppress non-maximal bounding boxes 
        extracted_bounding_boxes = class_.non_max_suppression(extracted_bounding_boxes, thresh_obj, thresh_iou)

        # sort bounding boxes by image index
        extracted_bounding_boxes.sort(key = lambda x : x[0])

        # convert to numpy array : (N * S * S, 7)
        extracted_bounding_boxes = numpy.float32(extracted_bounding_boxes)

        return extracted_bounding_boxes 

class YoloMetrics:
    def __init__(self, S : int, C : int, lambda_coord : float, lambda_noobj : float, thresh_obj : float, thresh_iou : float) -> None:
        assert isinstance(S, int)
        assert isinstance(C, int)
        assert isinstance(lambda_coord, float)
        assert isinstance(lambda_noobj, float)
        assert isinstance(thresh_obj, float)
        assert isinstance(thresh_iou, float)
        (self.S, self.C, self.lambda_coord, self.lambda_noobj, self.thresh_obj, self.thresh_iou) = (
            S, C, lambda_coord, lambda_noobj, thresh_obj, thresh_iou
        )

    def loss(self, y_true : EagerTensor, y_pred : EagerTensor) -> EagerTensor:
        """
            Parameters:
                [ 1 ] y_true : (N, S, S, C +  5)
                [ 2 ] y_pred : (N, S, S, C + 10)
            Limitations:
                [ 1 ] class probabilities : Slice[0,     C    )
                [ 2 ] objectness scores   : Slice[C,     C + 1), Slice[C + 5, C +  6)
                [ 3 ] (xywh) coordinates  : Slice[C + 1, C + 5), Slice[C + 6, C + 10)
            Suggestions:
                [ 1 ] class probabilities have been processed by Softmax
                [ 2 ] (xywh) coordinates are unbounded 
        """

        # class_loss : (,) [ calculate the classification loss of cells containing objects] [ (N, S, S, C) -> (N, S, S) -> (N, S, S, 1) -> (N, ) -> (,) ]
        class_loss = tf.reduce_mean(tf.reduce_sum(y_true[..., self.C : self.C + 1] * tf.expand_dims(tf.reduce_sum(tf.square(
                y_true[..., 0 : self.C] - y_pred[..., 0 : self.C]
            ), axis = 3), axis = 3), axis = [1, 2, 3]))

        # noobj_loss : (,) [ calculate the objectness loss of cells containing no objects ] [ (N, S, S, 1) -> (N, ) -> (,) ]
        noobj_loss = self.lambda_noobj * tf.reduce_mean(tf.reduce_sum((1 - y_true[..., self.C : self.C + 1]) * (tf.square(
                y_true[..., self.C : self.C + 1] - y_pred[..., self.C : self.C + 1]
            ) + tf.square(
                y_true[..., self.C : self.C + 1] - y_pred[..., self.C + 5 : self.C + 6]
            )), axis = [1, 2, 3]))

        # y_true : (N, S, S, 5) [ remove class probabilities ]
        y_true = y_true[..., self.C : self.C + 5]

        # y_pred : (N, S, S, 2, 5) [ remove class probabilities and reshape the bounding boxes ]
        y_pred = tf.reshape(y_pred[..., self.C : self.C + 10], (-1, self.S, self.S, 2, 5))

        # y_pred : (N, S, S, 5) [ extract the bounding boxes in each cell with greater iou scores ]
        # y_ious : (N, S, S, 1) [ extract the iou scores of the best bounding boxes ]
        y_pred, y_ious = MetricUtils.reduce_bounding_boxes(y_true, y_pred)

        # xy_loss : (,) [ calculate x-y loss of cells containing objects ] [ (N, S, S, 2) -> (N, S, S) -> (N, S, S, 1) -> (N, ) -> (,) ]
        xy_loss = self.lambda_coord * tf.reduce_mean(tf.reduce_sum(y_true[..., 0:1] * tf.expand_dims(tf.reduce_sum(tf.square(
                y_true[..., 1:3] - y_pred[..., 1:3]
            ), axis = 3), axis = 3), axis = [1, 2, 3]))

        # wh_loss : (,) [ calculate width-height loss of cells containing objects ] [ (N, S, S, 2) -> (N, S, S) -> (N, S, S, 1) -> (N,) -> (,) ]
        wh_loss = self.lambda_coord * tf.reduce_mean(tf.reduce_sum(y_true[..., 0:1] * tf.expand_dims(tf.reduce_sum(tf.square(
                tf.sqrt(y_true[..., 3:5] + epsilon()) - tf.sign(y_pred[..., 3:5]) * tf.sqrt(tf.abs(y_pred[..., 3:5]) + epsilon())
            ), axis = 3), axis = 3), axis = [1, 2, 3]))

        # obj_loss : (,) [ calculate objectness loss of cells containing objects ] [ (N, S, S, 1) -> (N, ) -> (,) ] 
        obj_loss = tf.reduce_mean(tf.reduce_sum(y_true[..., 0:1] * tf.square(y_true[..., 0:1] - y_pred[..., 0:1]), axis = [1, 2, 3]))
        #obj_loss = tf.reduce_mean(y_true[..., 0:1] * tf.square(y_true[..., 0:1] * y_ious[..., 0:1] - y_pred[..., 0:1]))

        # total_loss : (,) [ summation of localization (xywh), objectness and classification loss ]
        total_loss = xy_loss + wh_loss + obj_loss + noobj_loss + class_loss

        return total_loss 

    def mean_average_precision(self, y_true : EagerTensor, y_pred : EagerTensor) -> float:
        """
            Parameters:
                [ 1 ] y_true : (N, S, S, 25)
                [ 2 ] y_pred : (N, S, S, 30)
        """

        (true_bounding_boxes, pred_bounding_boxes) = (
            MetricUtils.extract_bounding_boxes(y_true, self.S, self.C, False).numpy().tolist(),
            MetricUtils.extract_bounding_boxes(y_pred, self.S, self.C, True).numpy().tolist()
        )

        # pred_bounding_boxes : { (img_idx, obj, cls_idx, x, y, w, h) }
        pred_bounding_boxes = MetricUtils.non_max_suppression(pred_bounding_boxes, self.thresh_obj, self.thresh_iou)

        true_bounding_boxes = sorted(filter(lambda x : x[1] >= self.thresh_obj, true_bounding_boxes), key = lambda x : x[1], reverse = True)

        ground_truth_history = [ 0 ] * len(true_bounding_boxes)

        # true_bounding_boxes : { (gnd_idx, img_idx, obj, cls_idx, x, y, w, h) }
        true_bounding_boxes = list(map(lambda x : [ x[0], *x[1] ], enumerate(true_bounding_boxes)))

        average_precisions = []

        for class_idx in range(self.C):

            (pred_boxes, true_boxes) = (
                list(filter(lambda x : x[2] == class_idx, pred_bounding_boxes)),
                list(filter(lambda x : x[3] == class_idx, true_bounding_boxes))
            )

            if (true_boxes.__len__() == 0):
                continue

            (true_positive, false_positive) = (
                numpy.zeros(len(pred_boxes)), numpy.zeros(len(pred_boxes))
            )

            for pred_idx, pred_box in enumerate(pred_boxes):

                __true_boxes = list(filter(lambda x : x[1] == pred_box[0], true_boxes))

                if not (__true_boxes):
                    false_positive[pred_idx] = 1
                    continue

                (best_idx, best_iou) = (0, -1)

                for true_box_idx, true_box in enumerate(__true_boxes):

                    temp_iou = MetricUtils.list_intersection_over_union(true_box[-4:], pred_box[-4:])

                    if (best_iou < temp_iou):
                        (best_idx, best_iou) = (true_box_idx, temp_iou)

                if ((best_iou < self.thresh_iou) or (ground_truth_history[__true_boxes[best_idx][0]])):
                    false_positive[pred_idx] = 1
                    continue 

                true_positive[pred_idx] = 1
                ground_truth_history[__true_boxes[best_idx][0]] = 1

            (TP_cumsum, FP_cumsum) = (
                numpy.cumsum(true_positive, axis = 0),
                numpy.cumsum(false_positive, axis = 0)
            )
            (precisions, recalls) = (
                numpy.concatenate([ [ 1 ], TP_cumsum / (TP_cumsum + FP_cumsum    + epsilon()) ]),
                numpy.concatenate([ [ 0 ], TP_cumsum / (len(true_boxes) + epsilon()) ])
            )
            average_precisions.append(numpy.trapz(precisions, recalls))

        return sum(average_precisions) / len(average_precisions)
    