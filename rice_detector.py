import cv2
import argparse
import os
import glob
import yolo_detector as yolo

ap = argparse.ArgumentParser()
ap.add_argument('-p', '--path', required=True,
                help = 'path to an input image file or directory')
ap.add_argument('-c', '--config', required=True,
                help = 'path to yolo config file')
ap.add_argument('-w', '--weights', required=True,
                help = 'path to yolo pre-trained weights')
ap.add_argument('-cl', '--classes', required=True,
                help = 'path to text file containing class names')
ap.add_argument('-cf', '--confidence', required=False, default = 0.4, type=float,
                help = 'confidence threshold value')
ap.add_argument('-nms', '--nms', required=False, default = 0.4, type=float,
                help = 'Non-Maximal-Suppression threshold value')
args = ap.parse_args()

# detector = yolo.YoloDetector(args.weights, args.config, args.classes)
detector = yolo.YoloDetector()
detector.load_net(args.weights, args.config, args.classes)
detector.set_model()

files = []
if os.path.isfile(args.path):
    files = [args.path]
elif os.path.isdir(args.path):
    exts = ['*.jpg', '*.png']
    files = [f for ext in exts for f in glob.glob(os.path.join(args.path, ext))]

for file in files:
    image = cv2.imread(file)

    # labels = detector.detect_and_label(image, args.confidence, args.nms)
    classes, scores, boxes = detector.detect(image, args.confidence, args.nms)
    labels = detector.label(image)

    print(file)
    path = os.path.splitext(file)
    if os.path.isfile(path[0]+'.txt'):
        txtfile = open(path[0]+'.txt','r')
        lines = txtfile.readlines()
        actual = []
        is_match = True
        for line in lines:
            classid = int(line.split(' ')[0])
            if len(classes) <=0 or classid != classes[0]:
                is_match = False
            actual.append(detector.class_name[classid])
        is_match = is_match and len(actual) == len(labels)
        print('Actual: ', actual, ' Match: ', is_match)

    print("Predict: ", labels)

    if len(files) > 1:
        print()
        continue

    winname = "rice detection"
    cv2.imshow(winname, image)
    while(True):
        # if cv2.waitKey(1) != -1 or \
        #    cv2.getWindowProperty(winname, cv2.WND_PROP_VISIBLE) < 1:
        #     break

        if any([cv2.waitKey(1) != -1,
                cv2.getWindowProperty(winname,cv2.WND_PROP_VISIBLE) < 1]):
            break
    cv2.imwrite("rice-detect-output.jpg", image)
    cv2.destroyAllWindows()
