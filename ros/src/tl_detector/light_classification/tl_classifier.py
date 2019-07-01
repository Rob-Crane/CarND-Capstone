from styx_msgs.msg import TrafficLight
import cv2

MODEL_PATH = r'light_classification/model.pb'
TRAIN_IMG_DIR = r'/media/sf_Virtualbox/sim_train/'
THRESHOLD = 0.5

class TLClassifier(object):
    def __init__(self):
        self.image_counter = 0
        self.graph = tf.Graph()
        with self.graph.as_default():
            graph_def = tf.GraphDef()

            with tf.gfile.GFile(MODEL_PATH, 'rb') as f:
                graph_def.ParseFromString(f.read())
                tf.import_graph_def(graph_def, name='')

            self.det_boxes = self.graph.get_tensor_by_name('detection_boxes:0')
            self.det_scores = self.graph.get_tensor_by_name('detection_scores:0')
            self.det_classes = self.graph.get_tensor_by_name('detection_classes:0')
            self.image_tensor = self.graph.get_tensor_by_name('image_tensor:0')
            self.num_detections = self.graph.get_tensor_by_name(
                'num_detections:0')

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        with self.graph.as_default():
            flattened = np.expand_dims(image, axis=0)
            (_, scores, classes, _) = self.sess.run(
                [self.boxes, self.scores, self.classes, self.num_detections],
                feed_dict={self.image_tensor: flattened})
        scores = np.squeeze(scores)
        classes = np.squeeze(classes).astype(np.int32)
        if scores[0] > self.threshold:
            if classes[0] == 1:
                return TrafficLight.GREEN
            elif classes[0] == 2:
                return TrafficLight.RED
            elif classes[0] == 3:
                return TrafficLight.YELLOW
        return TrafficLight.UNKNOWN

    def save_image(self, image, state):
        img_file = TRAIN_IMG_DIR + '{}_{}.png'.format(self.image_counter, state)
        print("Saving: ", img_file)
        self.image_counter += 1
        cv2.imwrite(img_file, image)
