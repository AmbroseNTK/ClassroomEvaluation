import json
from object_detection.utils import visualization_utils as vis_util
from object_detection.utils import label_map_util
from object_detection.utils import ops as utils_ops
from PIL import Image
from matplotlib import pyplot as plt
from io import StringIO
from collections import defaultdict
from distutils.version import StrictVersion
import zipfile
import tensorflow as tf
import tarfile
import six.moves.urllib as urllib
import os
import numpy as np
import sys
import shutil
import cv2
import logger
import sys

sys.path.append("/models/research")


if StrictVersion(tf.__version__) < StrictVersion('1.12.0'):
    raise ImportError(
        'Please upgrade your TensorFlow installation to v1.12.*.')

class BehaviorDetection:
# What model to download.
    MODEL_NAME = ''
#MODEL_FILE = MODEL_NAME + '.tar.gz'
#DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
    PATH_TO_FROZEN_GRAPH = MODEL_NAME + \
        'inference_graph/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
    PATH_TO_LABELS = "labelmap.pbtxt"

    session_id = "s0" # sys.argv[1]

    input_dir = "video.mp4" # sys.argv[2]

    result_folder = ["frames", "behaviors", "facial", "movement", "logs"]

    is_aborted = False

    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    category_index = label_map_util.create_category_index_from_labelmap(
        PATH_TO_LABELS, use_display_name=True)


    def load_image_into_numpy_array(self, image):
        (im_width, im_height) = image.size
        return np.array(image.getdata()).reshape(
            (im_height, im_width, 3)).astype(np.uint8)


    def run_inference_for_single_image(self, image, graph):
        with graph.as_default():
            with tf.Session() as sess:
            # Get handles to input and output tensors
                ops = tf.get_default_graph().get_operations()
                all_tensor_names = {
                    output.name for op in ops for output in op.outputs}
                tensor_dict = {}
                for key in [
                    'num_detections', 'detection_boxes', 'detection_scores',
                    'detection_classes', 'detection_masks'
                ]:
                    tensor_name = key + ':0'
                    if tensor_name in all_tensor_names:
                        tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
                        tensor_name)
                if 'detection_masks' in tensor_dict:
                # The following processing is only for single image
                    detection_boxes = tf.squeeze(
                        tensor_dict['detection_boxes'], [0])
                    detection_masks = tf.squeeze(
                        tensor_dict['detection_masks'], [0])
                # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
                    real_num_detection = tf.cast(
                        tensor_dict['num_detections'][0], tf.int32)
                    detection_boxes = tf.slice(detection_boxes, [0, 0], [
                                           real_num_detection, -1])
                    detection_masks = tf.slice(detection_masks, [0, 0, 0], [
                                           real_num_detection, -1, -1])
                    detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                        detection_masks, detection_boxes, image.shape[1], image.shape[2])
                    detection_masks_reframed = tf.cast(
                        tf.greater(detection_masks_reframed, 0.5), tf.uint8)
                # Follow the convention by adding back the batch dimension
                    tensor_dict['detection_masks'] = tf.expand_dims(
                        detection_masks_reframed, 0)
                image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

            # Run inference
                output_dict = sess.run(tensor_dict,
                                   feed_dict={image_tensor: image})

            # all outputs are float32 numpy arrays, so convert types as appropriate
                output_dict['num_detections'] = int(
                    output_dict['num_detections'][0])
                output_dict['detection_classes'] = output_dict[
                    'detection_classes'][0].astype(np.int64)
                output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
                output_dict['detection_scores'] = output_dict['detection_scores'][0]
                if 'detection_masks' in output_dict:
                    output_dict['detection_masks'] = output_dict['detection_masks'][0]
        return output_dict

    def behaviors_detect(self,session_id):

        TEST_IMAGE_PATHS = os.listdir("result/"+session_id+"/frames/")
        count = 0

        for image_path in TEST_IMAGE_PATHS:
            if self.is_aborted:
                return 0
            image = Image.open("result/"+session_id+"/frames/"+image_path)
    # the array based representation of the image will be used later in order to prepare the
    # result image with boxes and labels on it.
            image_np = self.load_image_into_numpy_array(image)
    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
            image_np_expanded = np.expand_dims(image_np, axis=0)
    # Actual detection.
            output_dict = self.run_inference_for_single_image(
                image_np_expanded, self.detection_graph)

            obj = {
                "boxes": output_dict['detection_boxes'].tolist(),
                "classes": output_dict['detection_classes'].tolist(),
                "scores": output_dict['detection_scores'].tolist()
            }

            result = open("result/"+session_id+"/behaviors/"+image_path+'.json', 'w')
            json.dump(obj, result)
            result.close()
            print("Behaviors: " + image_path + " done")
            count += 1
        #logger.write_log(logger.LOG_BEHAVIOR, session_id,
        #             count, len(TEST_IMAGE_PATHS))
