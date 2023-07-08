import argparse
import os.path
import sys
import logging
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))

from Config import cfg
from Config import update_config
from datetime import datetime

from SLPT import Sparse_alignment_network

import torch, cv2, math
import numpy as np
import pprint

import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
import Face_Detector
import utils
import shutil

def parse_SLPT_args():
    parser = argparse.ArgumentParser(description='Video Demo')

    # face detector, Fixed
    parser.add_argument('--confidence_threshold', default=0.7, type=float, help='confidence_threshold')
    parser.add_argument('--top_k', default=5000, type=int, help='top_k')
    parser.add_argument('--nms_threshold', default=0.3, type=float, help='nms_threshold')
    parser.add_argument('--keep_top_k', default=750, type=int, help='keep_top_k')
    parser.add_argument('--vis_thres', default=0.3, type=float, help='visualization_threshold')
    parser.add_argument('--base_layers', default=16, type=int, help='the number of the output of the first layer')
    parser.add_argument('--device', default='cuda:0', help='which device the program will run on. cuda:0, cuda:1, ...')

    # Fixed, no useful
    parser.add_argument('--modelDir', help='model directory', type=str, default='./Weight')
    parser.add_argument('--checkpoint', help='checkpoint file', type=str, default='WFLW_6_layer.pth')
    parser.add_argument('--logDir', help='log directory', type=str, default='./log')
    parser.add_argument('--dataDir', help='data directory', type=str, default='./')
    parser.add_argument('--prevModelDir', help='prev Model directory', type=str, default=None)

    SLPT_args = parser.parse_args(args=[])

    return SLPT_args


def draw_landmark(landmark, image):

    for (x, y) in (landmark + 0.5).astype(np.int32):
        cv2.circle(image, (x, y), 3, (255,0, 0), -1)

    return image


def crop_img(img, bbox, transform):
    x1, y1, x2, y2 = (bbox[:4] + 0.5).astype(np.int32)

    w = x2 - x1 + 1
    h = y2 - y1 + 1
    cx = x1 + w // 2
    cy = y1 + h // 2
    center = np.array([cx, cy])

    scale = max(math.ceil(x2) - math.floor(x1),
                math.ceil(y2) - math.floor(y1)) / 200.0

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    input, trans = utils.crop_v2(img, center, scale * 1.15, (256, 256))

    input = transform(input).unsqueeze(0)

    return input, trans

def face_detection(img, model, im_width, im_height):
    args = parse_SLPT_args() 
    device = torch.device(args.device) 
    img = cv2.resize(img, (320, 240), interpolation=cv2.INTER_NEAREST)
    img = np.float32(img)
    img = img.transpose(2, 0, 1)
    img = torch.from_numpy(img).unsqueeze(0)
    img = img.to(device)

    scale = torch.Tensor([im_width, im_height, im_width, im_height,
                          im_width, im_height, im_width, im_height,
                          im_width, im_height, im_width, im_height,
                          im_width, im_height])
    scale = scale.to(device)

    # feed forward
    loc, conf, iou = model(img)

    # post processing
    priorbox = Face_Detector.PriorBox(Face_Detector.cfg, image_size=(im_height, im_width))
    priors = priorbox.forward()
    priors = priors.to(device)
    prior_data = priors.data
    boxes = Face_Detector.decode(loc.data.squeeze(0), prior_data, Face_Detector.cfg['variance'])
    boxes = boxes * scale
    boxes = boxes.cpu().numpy()
    cls_scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
    iou_scores = iou.squeeze(0).data.cpu().numpy()[:, 0]
    # clamp here for the compatibility for ONNX
    _idx = np.where(iou_scores < 0.)
    iou_scores[_idx] = 0.
    _idx = np.where(iou_scores > 1.)
    iou_scores[_idx] = 1.
    scores = np.sqrt(cls_scores * iou_scores)

    # ignore low scores
    inds = np.where(scores > args.confidence_threshold)[0]
    boxes = boxes[inds]
    scores = scores[inds]

    # keep top-K before NMS
    order = scores.argsort()[::-1][:args.top_k]
    boxes = boxes[order]
    scores = scores[order]

    # do NMS
    dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
    selected_idx = np.array([0, 1, 2, 3, 14])
    keep = Face_Detector.nms(dets[:, selected_idx], args.nms_threshold)
    dets = dets[keep, :]

    # keep top-K faster NMS
    dets = dets[:args.keep_top_k, :]

    return dets

def find_max_box(box_array):
    potential_box = []
    for b in box_array.copy():
        if b[14] < 0.3:
            continue
        potential_box.append(np.array([b[0], b[1], b[2], b[3], b[14]]).astype('int'))

    if len(potential_box) > 0:
        x1, y1, x2, y2 = (potential_box[0][:4]).astype(np.int32)
        Max_box = (x2 - x1) * (y2 - y1)
        Max_index = 0
        for index in range(1, len(potential_box)):
            x1, y1, x2, y2 = (potential_box[index][:4]).astype(np.int32)
            temp_box = (x2 - x1) * (y2 - y1)
            if temp_box >= Max_box:
                Max_box = temp_box
                Max_index = index
        return box_array[Max_index].copy()
    else:
        return None


class SLPT_Detector():
    def __init__(self):
        args = parse_SLPT_args()
        update_config(cfg, args)
        device = torch.device(args.device)
        torch.set_grad_enabled(False)

        # Cuda
        cudnn.benchmark = True
        torch.backends.cudnn.enabled = True

        face_detector_ckpt_path = "SLPT_dev/Weight/Face_Detector/yunet_final.pth"
        face_alignment_ckpt_path = "SLPT_dev/Weight/WFLW_6_layer.pth"

        # load face detector
        net = Face_Detector.YuFaceDetectNet(phase='test', size=None)  # initialize detector
        net = Face_Detector.load_model(net, face_detector_ckpt_path, True)
        net.eval()
        net = net.to(device)
        self.net = net
        print('Finished loading Face Detector!')

        model = Sparse_alignment_network(cfg.WFLW.NUM_POINT, cfg.MODEL.OUT_DIM,
                                        cfg.MODEL.TRAINABLE, cfg.MODEL.INTER_LAYER,
                                        cfg.MODEL.DILATION, cfg.TRANSFORMER.NHEAD,
                                        cfg.TRANSFORMER.FEED_DIM, cfg.WFLW.INITIAL_PATH, cfg)
        model = torch.nn.DataParallel(model, device_ids=cfg.GPUS).cuda()

        checkpoint = torch.load(face_alignment_ckpt_path)
        pretrained_dict = {k: v for k, v in checkpoint.items()
                        if k in model.module.state_dict().keys()}
        model.module.load_state_dict(pretrained_dict)
        model.eval()
        self.model = model
        print('Finished loading face landmark detector')

        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])

    def detect_face(self, img):
        """
            detect the faces from the original img, and get the max box with filter rules
            Args:
                img: np.array, the original image
            Returns:
                max_bbox: np.array, the max box of the face
                dets: np.array, all results for face detection, numx14: box(4), landmarks(5*2), confidence(1)
        """
        dets = face_detection(img.copy(), self.net, 320, 240)
        max_bbox = find_max_box(dets)
        return max_bbox, dets


    def adjust_boxes(self, img, bbox):
        """
            adjust the bbox to the max value of height and width, and crop the image
            Args:
                img: np.array, the original image
                bbox: np.array, the max box of the face
            Returns:
                alignment_input: np.array, the cropped image
                trans: np.array, the transform matrix    
        """
        if bbox is not None: 
            bbox[0] = int(bbox[0] / 320.0 * img.shape[1] + 0.5)
            bbox[2] = int(bbox[2] / 320.0 * img.shape[1] + 0.5)
            bbox[1] = int(bbox[1] / 240.0 * img.shape[0] + 0.5)
            bbox[3] = int(bbox[3] / 240.0 * img.shape[0] + 0.5)
            alignment_input, trans = crop_img(img.copy(), bbox, self.normalize)
            return alignment_input, trans
        else:
            raise ValueError("No face detected in the image.")
        

    def detect_face_via_landmarks(self, input_img, trans, confidence):
        """
            model detects landmarks according the crop img and calculate the bounding box via the max and min values
            Args:
                input_img: ndarray, the input image (opencv format)
                trans: ndarray, the transformation matrix
                confidence: float, the confidence of the face detection
            Returns:
                box_info: dict, the bounding box information, including the box (left-top point and wh), confidence, keypoints and 98 landmarks
        """
        outputs_initial = self.model(input_img.cuda())
        output = outputs_initial[2][0, -1, :, :].cpu().numpy()

        landmark = utils.transform_pixel_v2(output * cfg.MODEL.IMG_SIZE, trans, inverse=True)
        right_bottom = landmark.max(0)
        left_top = landmark.min(0)

        box_info = {
            'box' : [round(left_top[0]), round(left_top[1]), round(right_bottom[0]-left_top[0]), round(right_bottom[1]-left_top[1])],
            'confidence' : confidence,
            'keypoints' : {
                'nose': (int(landmark[54][0]), int(landmark[54][1])),
                'mouth_left': (int(landmark[88][0]), int(landmark[88][1])),
                'mouth_right': (int(landmark[92][0]), int(landmark[92][1])),
                'right_eye': (int(landmark[72][0]), int(landmark[72][1])),
                'left_eye': (int(landmark[60][0]), int(landmark[60][1]))
            },
            'landmark98': landmark
        }
        return box_info

    def detect_face_from_opencv_img(self, img):
        """
            The whole pipeline for detect face from opencv img (BGR)
            This function is used for video detection
            Args:
                img: np.ndarray, opencv img
            Returns:
                fine_bbox: dict, bbox information contains box (left-top point and wh) and landmark98, None if no face detected
        """
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # test rgb or bgr
        coarse_bbox, coarse_all_dets = self.detect_face(img)
        if coarse_bbox is None:
            return None
        crop_img, trans = self.adjust_boxes(img, coarse_bbox)
        fine_bbox = self.detect_face_via_landmarks(crop_img, trans, coarse_bbox[14])
        fine_bbox.update({"coarse_bbox":coarse_bbox, 'coarse_all_dets':coarse_all_dets})
        return fine_bbox
    
def is_logging_configured():
    """
        This function checks if the logging is configured.
        Parameters:
            None
        Returns:
            - True if at least one logging handler is configured.
            - False if no logging handlers are configured.

    """
    handlers = logging.getLogger().handlers
    return bool(handlers)

def init_Logger(folder_path):
    """
        Initializes the logger for logging output.
        Parameters:
            folder_path: str, the path to the folder where the log file will be stored.
        Returns:
            None
    """
    folder_name = "_".join(folder_path.split('/'))
    time_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file_path = os.path.join("output_log", f'{folder_name}_output_{time_str}.log')
    dirname = os.path.dirname(log_file_path)
    os.makedirs(dirname,exist_ok=True)
    logging.basicConfig(filename=log_file_path, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', force=True)
    

class SLPT_Tooler():
    def __init__(self):
        self.slpt = SLPT_Detector()
    
    def draw_result(self, vis_img, fine_box):
        """
            Draw face boxes(coarse_bbox and fine_box) and landmarks
            Args:
                vis_img: np.ndarray, opencv img
                fine_box: dict, bbox information contains box, coarse_all_dets, coarse_bbox, img_path and landmark98, None if no face detected
        """
        left_top = fine_box['box'][:2]
        right_bottom = [left_top[0] + fine_box['box'][2], left_top[1] + fine_box['box'][3]]
        cv2.rectangle(vis_img, left_top, right_bottom, (0, 255, 255), 2)

        if "coarse_bbox" in fine_box.keys():
            coarse_box = fine_box['coarse_bbox']
            coarse_left_top = [int(coarse_box[0]), int(coarse_box[1])]
            coarse_right_bottom = [int(coarse_box[2]), int(coarse_box[3])]
            cv2.rectangle(vis_img, coarse_left_top, coarse_right_bottom, (0, 0, 255), 2)

        draw_landmark(fine_box['landmark98'], vis_img)

    def vis_result_from_img(self, root_folder, fine_box):
        """
            Visualize single image (including face boxes and landmarks)
            Args:
                root_folder: str, output root dir
                fine_box: dict, bbox information contains box, coarse_all_dets, coarse_bbox, img_path and landmark98, None if no face detected
        """
        if fine_box is None:
            print("Warning: box information is empty!")
            return
        img_path = fine_box['img_path']
        img = cv2.imread(img_path)
        assert img is not None, f"Error: Failed to load image from '{img_path}'."

        file_name = os.path.basename(img_path)
        output_folder = os.path.join(root_folder, os.path.dirname(img_path))
        os.makedirs(output_folder, exist_ok=True)
        output_path = os.path.join(output_folder, file_name)

        vis_img = img.copy()
        self.draw_result(vis_img, fine_box) 
        cv2.imwrite(output_path, vis_img)

    def vis_results_from_folder(self, root_folder, results):
        """
            Visualize the folder image (including facial boxes, largest box and landmarks)
            Args:
                root_folder: str, output root dir
                results: list contains dictionary（bbox information contains box，coarse_all_dets, coarse_bbox, img_path and landmark98, None if no face detected）
        """
        for bbox in results:
            if bbox is None:
                continue
            img_path = bbox['img_path']
            output_folder = os.path.join(root_folder, os.path.dirname(img_path))
            os.makedirs(output_folder, exist_ok=True)
            output_path = os.path.join(output_folder, os.path.basename(img_path))

            vis_img = cv2.imread(img_path)
            self.draw_result(vis_img, bbox)
            cv2.imwrite(output_path, vis_img)

    def vis_coarse_all_dets_for_one_image(self, root_folder, fine_bbox):
        """
            Visualize all detected facial box, largest box and landmarks.
            here fine_bbox is not None but a dict
            Args:
                root_folder: str, output root dir
                fine_bbox: list contains dictionary（bbox information contains box, coarse_all_dets, coarse_bbox, img_path and landmark98, None if no face detected）
        """

        img_path = fine_bbox['img_path']
        img = cv2.imread(img_path)

        file_name = os.path.basename(img_path)
        output_folder = os.path.join(root_folder, os.path.dirname(img_path))
        os.makedirs(output_folder, exist_ok=True)
        output_path = os.path.join(output_folder, file_name)

        vis_img = img.copy()

        coarse_all_dets = fine_bbox['coarse_all_dets']

        # vis all faces and their confidences
        if len(coarse_all_dets) > 1:
            for bbox_det in coarse_all_dets:
                bbox = bbox_det[:4].copy()
                bbox[0] = bbox[0] / 320.0 * vis_img.shape[1] + 0.5
                bbox[2] = bbox[2] / 320.0 * vis_img.shape[1] + 0.5
                bbox[1] = bbox[1] / 240.0 * vis_img.shape[0] + 0.5
                bbox[3] = bbox[3] / 240.0 * vis_img.shape[0] + 0.5
                bbox = bbox.astype(np.int32)

                coarse_left_top = (bbox[0], bbox[1])
                coarse_right_bottom = (bbox[2], bbox[3])
                cv2.rectangle(vis_img, coarse_left_top, coarse_right_bottom, (0, 255, 0), 2)
                text_pos = (bbox[0], bbox[1] + 10)
                cv2.putText(vis_img, f"{bbox_det[14]:.2f}", text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)
        
        # max face (red) will replace the previous face box (green)
        self.draw_result(vis_img, fine_bbox)
        cv2.imwrite(output_path, vis_img)
        
    def vis_warning_result_from_one_result(self, fine_bbox, img_path):
        """
            Visualize all detected facial box, largest box and landmarks.
            Args:
                fine_bbox: list contains dictionary（bbox information contains box, coarse_all_dets, coarse_bbox, img_path and landmark98, None if no face detected）
                img_path: dir, input dir( fine_bbox is none, without image path)
        """
        warn_result_path = "warning_result"
        os.makedirs(warn_result_path, exist_ok=True)

        # when detect no faces, copy img to warn result folder
        if fine_bbox is None:
            # save naive image into the warning result, in function: vis warning result from that img
            shutil.copy(img_path, warn_result_path)
        else:
            num_box = len(fine_bbox["coarse_all_dets"])

            if num_box > 1:
                # when all_dets contains multiple faces
                self.vis_coarse_all_dets_for_one_image(warn_result_path, fine_bbox)
            else:
                # when aspect_r is wrong
                self.vis_result_from_img(warn_result_path, fine_bbox)
    
    def detect_vis_face_from_video(self, root_dir, video_path):
        """
            draw single box from video frame
            It will save frames in root_dir/video_path/
            Args:
                root_dir: str, output root dir
                video_path: str, input video path
        """
        cap = cv2.VideoCapture(video_path)
        im_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        im_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        video_name = os.path.basename(video_path)[:-4]
        output_folder = os.path.join(root_dir, video_name)
        os.makedirs(output_folder, exist_ok=True)
        output_path = os.path.join(output_folder, "output_video.mp4")

        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        out = cv2.VideoWriter(output_path, fourcc, 20.0, (im_width, im_height))

        ret, frame = cap.read()
        frame_count = 0
        while ret:
            bbox = self.slpt.detect_face_from_opencv_img(frame)
            if bbox is None:
                continue
            vis_img = frame.copy()
            left_top = bbox['box'][:2]
            right_bottom = [left_top[0] + bbox['box'][2], left_top[1] + bbox['box'][3]]
            cv2.rectangle(vis_img, left_top, right_bottom, (0, 255, 255), 2)
            draw_landmark(bbox['landmark98'], vis_img)

            out.write(vis_img)
            cv2.imwrite(os.path.join(output_folder, f"vis_frame_{frame_count}.jpg"), vis_img)

            frame_count += 1
            ret, frame = cap.read()

        cap.release()
        out.release()
        cv2.destroyAllWindows()

    def detect_faces_from_folder_recur(self, folder_path, warn_record=False):
        """
            Recursive detection of folder images,base recursive invocation
            all_results may contains None value, which means no face detected
            Args:
                folder_path: str, input folder path
                warn_record: bool, provid warning information in log
            return:
                all_results: list, imgs' bbox information contains box and landmark98
        """
        if not is_logging_configured():
            init_Logger(folder_path)
        all_results = []

        def get_all_folders(folder_path):
            folders = [f.path for f in os.scandir(folder_path) if f.is_dir()]
            return folders + [subfolder for folder in folders for subfolder in get_all_folders(folder)]

        all_folders = get_all_folders(folder_path)
        for one_folder in all_folders:
            all_results += self.detect_faces_from_one_folder(one_folder, warn_record)
        return all_results

    def detect_faces_from_one_folder(self, folder_path, warn_record=False):
        """
            Detect one max face according to the image from one folder
            folder_results may contains None value, which means no face detected
            Args:
                folder_path: str, input folder path
                warn_record: bool, provid warning information in log
            Returns:
                folder_results: list, contains bbox info
        """
        if not is_logging_configured():
            init_Logger(folder_path)

        folder_results = []
        for filename in os.listdir(folder_path):
            if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".jpeg"):
                img_path = os.path.join(folder_path, filename)
                folder_results.append(self.detect_face_from_img_path(img_path, warn_record=warn_record))
        return folder_results

    def detect_face_from_img_path(self, img_path, warn_record=False):
        """
            The whole pipeline for detect face from img path
            Args:
                img_path: str, the path of the image
                warn_record: bool, provid warning information in log
            Returns:
                fine_bbox: dict, bbox information contains box and landmark98, None if no face detected
        """
        if not is_logging_configured():
            init_Logger(img_path)

        img = cv2.imread(img_path)
        if img is None or img.size == 0:
            logging.error(f"Failed to load image from '{img_path}'.")
            return None

        fine_bbox = self.slpt.detect_face_from_opencv_img(img) # here img is bgr

        if fine_bbox is not None:
            fine_bbox.update({"img_path":img_path})
        
        if warn_record:
            self.record_warning_from_single_img_process(fine_bbox, img_path)
        return fine_bbox

    def record_warning_from_single_img_process(self, fine_bbox, img_path):
        """
            Record error messages in log and visualize them
            Args:
                img_path: str, the path of the image
                fine_bbox: dict, bbox information, None if no face detected
        """
        if fine_bbox is None:
            logging.warning(f"No face detected in '{img_path}'.")
            self.vis_warning_result_from_one_result(fine_bbox, img_path)
            return

        coarse_all_dets = fine_bbox["coarse_all_dets"]
        num_box = len(coarse_all_dets)
        is_warn = False
        warn_txt = ""

        if num_box > 1 :
            is_warn = True
            warn_txt += f"detect {len(coarse_all_dets)} faces in {img_path}, and use the max box as the sole face result"

        if fine_bbox is not None:
            aspect_r = fine_bbox['box'][2] / fine_bbox['box'][3]
            if abs(aspect_r - 1) > 0.2:
                is_warn = True
                warn_txt += f"The aspect ratio of the face box is incorrect in '{img_path}'."

        # fine_bbox.update({"is_warn":is_warn})
        if not is_warn:
            logging.info(f"Successfully detected face in '{img_path}'.")
        else:
            logging.warning(warn_txt)
            self.vis_warning_result_from_one_result(fine_bbox, img_path)        

if __name__ == '__main__':
    img_path = "naive_data/try/for_mtcnn/image_003_1.jpg"
    folder_path = "naive_data/300W" # "naive_data/try"
    video_path = "36.mp4"

    slpt_detector = SLPT_Tooler()

    # # detect one image
    # # img = cv2.imread(img_path)
    # box_info = slpt_detector.detect_face_from_img_path(img_path, warn_record=True)
    # slpt_detector.vis_result_from_img("test_data",box_info)

    #detect one folder
    init_Logger(folder_path) # logger reset, if not, it will record in previous log file.
    results_correct = slpt_detector.detect_faces_from_folder_recur(folder_path, warn_record=True)
    slpt_detector.vis_results_from_folder("test_data", results_correct)
    
    # #detect one video
    # slpt_detector.detect_vis_face_from_video("test_data", video_path)
    