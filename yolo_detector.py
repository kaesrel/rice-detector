import cv2

class YoloDetector:
    COLORS = [(0, 255, 0), (0, 0, 255), (255, 0, 0),
              (255, 255, 0), (255, 0, 255), (0, 255, 255)]

    def __init__(self, weights_file=None, config_file=None, class_file=None, \
                 input_shape=(416,416), scale=1/255):
        self.net = None
        self.class_name = None
        self.model = None
        if weights_file != None and config_file != None and class_file != None:
            self.load_net(weights_file, config_file, class_file)
            self.set_model(input_shape, scale)
        self.classes = None
        self.scores = None
        self.boxes = None

    def load_net(self, weights_file, config_file, class_file):
        self.net = cv2.dnn.readNet(weights_file,config_file)
        # self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        # self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
        self.class_name = []
        with open(class_file, 'r') as f:
            self.class_name = [cname.strip() for cname in f.readlines()]

    def set_model(self, input_shape=(416,416), scale=1/255):
        if self.net is None or self.class_name == []:
            raise Exception('set_model(...) is failed : load_net(...) must be successfully called before (at least once).')
        self.model = cv2.dnn_DetectionModel(self.net)
        self.model.setInputParams(size=input_shape, scale=scale, swapRB=True)

    def detect(self, image, conf_threshold=0.4, nms_threshold=0.4):
        if self.model is None:
            raise Exception('detect(...) is failed : set_model(...) must be successfully called once before (at least once).')
        self.classes, self.scores, self.boxes = self.model.detect(image, conf_threshold, nms_threshold)
        return self.classes, self.scores, self.boxes

    def label(self, image):
        labels = []
        for (classid, score, box) in zip(self.classes, self.scores, self.boxes):
            color = self.COLORS[int(classid) % len(self.COLORS)]
            label = f'{self.class_name[classid]}: {score:0.4f}'
            # print(label)
            cv2.rectangle(image, box, color, 1)
            cv2.rectangle(image, (box[0], box[1] - 20), (box[0] + box[2], box[1]), color, -1)
            cv2.putText(image, label, (box[0]+5, box[1]-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255),2)
            labels.append(label)
        return labels

    def detect_and_label(self, image, conf_threshold=0.4, nms_threshold=0.4):
        self.detect(image, conf_threshold, nms_threshold)
        return self.label(image)
