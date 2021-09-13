import argparse
import cv2
import numpy as np
import sys
from numpy.core.fromnumeric import resize

def make_cat_passport_image(input_image_path, haar_model_path):

    # Read image
    img = cv2.imread(input_image_path)
    # Convert image to grayscale
    img_grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Normalize image intensity
    norm_img = np.zeros(img_grayscale.shape)
    norm_img = cv2.normalize(img_grayscale,norm_img,0, 255,cv2.NORM_MINMAX)
    # Resize image
    resized = cv2.resize(norm_img,(640,480), interpolation=cv2.INTER_AREA)
    # Detect cat faces using Haar Cascade
    detector = cv2.CascadeClassifier(haar_model_path)
    rects = detector.detectMultiScale(img, scaleFactor = 1.1, minNeighbors = 5, minSize = (75,75))
    
    # Draw bounding box
    for (i, (x, y, w, h)) in enumerate(rects): 
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2) 
        cv2.putText(img, "Cat #{}".format(i + 1), (x, y - 10), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
    # Display result image
    cv2.imshow("window_name", img) 
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # Crop image
    x, y, w, h = rects[0]
    image = img[y:y+h, x:x+w]
    # Save result image to file
    cv2.imwrite('out.jpg', image)
    return


def build_argparser():
    parser = argparse.ArgumentParser(
        description='Speech denoising demo', add_help=False)
    args = parser.add_argument_group('Options')
    args.add_argument('-h', '--help', action='help', default=argparse.SUPPRESS,
                      help='Show this help message and exit.')
    args.add_argument('-m', '--model', type=str, required=True,
                      help='Required. Path to .XML file with pre-trained model.')
    args.add_argument('-i', '--input', type=str, required=True,
                      help='Required. Path to input image')
    return parser


def main():
    
    args = build_argparser().parse_args()
    make_cat_passport_image(args.input, args.model)

    return 0


if __name__ == '__main__':
    sys.exit(main() or 0)
