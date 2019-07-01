from styx_msgs.msg import TrafficLight
import cv2

class TLClassifier(object):
    def __init__(self):
        #TODO load classifier
        self.image_counter = 4611
        pass

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        #TODO implement light color prediction
        return TrafficLight.UNKNOWN

    def save_image(self, image, state):
        img_file = '/media/sf_Virtualbox/sim_train/{}_{}.png'.format(self.image_counter, state)
        print("Saving: ", img_file)
        self.image_counter += 1
        cv2.imwrite(img_file, image)
