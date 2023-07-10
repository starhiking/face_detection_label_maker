# -*- coding:utf-8  -*-
# Auther : Lyu jiayi
# Data : 2023/06/08
# Comment: This is a face detection framework based on MTCNN or RetinaFace.
# Modified: 2021/06/08 V1->V2: complete basic functions;test V2&V3 module; image path in txt changed to relative path
# Modified: 2021/06/09 V2->V3: write comments and make the code elegent.

import argparse
import os
import sys
import cv2
from mtcnn import MTCNN
import numpy as np
sys.path.append('.')  # 将SLPT-master文件夹添加到sys.path中
from SLPT_dev.SLPT_detector import SLPT_Tooler



class FaceDetecter():
    def __init__(self, net_type="mtcnn", return_type="v1", scale=1.0):
        """
        Args:
            net_type: choice:[mtcnn,retinaface],default is mtcnn.
            return_type: choice:[v1,v2,v3], v1 is return left-top point and 宽高, v2 is return left-top and right-bottom points, v3 is return center point and 宽高.
            scale: float, default is 1.0. If larger than 1.0, enlarge the box with this ratio.
        """
        self.net_type = net_type
        self.return_type = return_type
        self.scale = scale

        # init network according the net_type
        if self.net_type == "mtcnn":
            self.net = MTCNN()
        elif self.net_type == "SLPT":
            self.net = SLPT_Tooler()
        else:
            raise ValueError("Not support network")

    def detect_img(self, img_path):
        """
        Args:
            img_path: str, img file path

        Return:
            box_info: list, box information according the return type
        """
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.net_type == "mtcnn":
            # detect face boxes
            boxes = self.net.detect_faces(img)
            # filter useless boxes with rules
            filtered_boxes = self.filter_boxes(boxes,img)
            # adjust box size according max width or height (maintain the wh ratio)
            adjusted_boxes = self.adjust_box_size(filtered_boxes, img.shape[1], img.shape[0])
            # return box information by return_type
            box_info = self.convert_to_return_type(adjusted_boxes)
            return box_info

    def filter_boxes(self, boxes, img):
        """
        Filter face box based on specific rules

        Args:
            boxes: img, boxes form detect_faces

        Return:
            filtered_boxes: list, 被选中的'box'、'confidence' 'keypoints'
        """
        filtered_boxes = []

        if len(boxes) != 0:

            box_areas = np.array([box['box'][2] * box['box'][3] for box in boxes]) / (img.shape[0] * img.shape[1])
            center_points = np.array(
                [(box['box'][0] + box['box'][2] // 2, box['box'][1] + box['box'][3] // 2) for box in boxes])
            center_differences = np.linalg.norm(center_points - np.array(img.shape[:2])[::-1] // 2, axis=1)
            confidences = np.array([box['confidence'] for box in boxes])
            idx_center_small = np.argsort(center_differences)
            idx_area_large = np.argsort(box_areas)[::-1]

            if idx_center_small[0] != idx_area_large[0]:
                if box_areas[idx_center_small[0]] > 0.15 and confidences[idx_center_small[0]] > 0.95:
                    max_largest_idx = idx_center_small[0]
                else:
                    max_largest_idx = idx_area_large[0]
            else:
                max_largest_idx = idx_center_small[0]

            filtered_boxes.append(boxes[max_largest_idx])
        else:
            height, width, _ = img.shape
            filtered_boxes.append({
                'box': [0, 0, width, height],  # 设置为图像边角的四个顶点坐标
                'confidence': 0.0,  # 设置置信度为0
                'keypoints': {}  # 设置空的关键点
            })

        return filtered_boxes


    def adjust_box_size(self, boxes, max_width, max_height):
        """
        resize the face box according to the scale

        Args:
            boxes: list, selected'box'、'confidence' 'keypoints'
            max_width, max_height

        Return:
            adjusted_boxes: list, enlarge box information by scale , and maintain its return type
                        'box'：tuple，represent the position and size of the face box (left-top_x,left-top_y,w,h)
                        'confidence'：confidence。
        """
        adjusted_boxes = []
        for box in boxes:
            box_info = box['box'] if 'box' in box else None
            if box_info is None or len(box_info) != 4:
                continue

            confidence = box['confidence']

            center_point = (int(box_info[0] + box_info[2] // 2), int(box_info[1] + box_info[3] // 2))

            edge_length = max(box_info[2], box_info[3])
            new_size = (int(edge_length * self.scale), int(edge_length * self.scale))
            right_bottom = (center_point[0] + new_size[0] // 2, center_point[1] + new_size[1] // 2)
            left_top = (center_point[0] - new_size[0] // 2, center_point[1] - new_size[1] // 2)
            right_bottom = (min(right_bottom[0], max_width), min(right_bottom[1], max_height))
            left_top = (max(left_top[0], 0), max(left_top[1], 0))
            adjusted_boxes.append(
                {'box': (left_top[0], left_top[1], right_bottom[0] - left_top[0], right_bottom[1] - left_top[1]),
                 'confidence': confidence})
        return adjusted_boxes


    def convert_to_return_type(self, adjusted_boxes):
        """
        return face box's information
        Args:
            adjusted_boxes: list, enlarge box information by scale , and maintain its return type，
                        'box'：tuple，reptresent the position and size of the face box (lefttop_x,lefttop_y,w,h)
                        'confidence'：confidence'
            return_type

        Return:
            retruns the representation that requires a face box based on x

        """
        box_info = []

        if self.return_type == "v1":
            # Convert to return type v1: left-top point and width, height
            box = adjusted_boxes[0]
            left_top = (box['box'][0], box['box'][1])
            width = box['box'][2]
            height = box['box'][3]
            # box_info = (left_top, width, height)
            box_info = (box['box'][0], box['box'][1], width, height)


        elif self.return_type == "v2":
            # Convert to return type v2: left-top and right-bottom points
            box = adjusted_boxes[0]
            left_top = (box['box'][0], box['box'][1])
            right_bottom = (box['box'][0] + box['box'][2], box['box'][1] + box['box'][3])
            # box_info = (left_top, right_bottom)
            box_info = (box['box'][0], box['box'][1],box['box'][0] + box['box'][2], box['box'][1] + box['box'][3])

        elif self.return_type == "v3":
            # Convert to return type v3: center point and width, height
            box = adjusted_boxes[0]
            center = (box['box'][0] + box['box'][2] // 2, box['box'][1] + box['box'][3] // 2)
            width = box['box'][2]
            height = box['box'][3]
            # box_info = (center, width, height)
            box_info = (box['box'][0] + box['box'][2] // 2, box['box'][1] + box['box'][3] // 2,box['box'][2], box['box'][3])

        return box_info

    def convert_lefttop_rightbottom(self, img_path, box_info):
        """
        left_top,right_bottom for visualization

        Args:
                   img_path: str, file path
                   box_info: list, box information according its return type

        returns:
                left_top,right_bottom for visualization
        """

        if self.return_type == "v1":
            left, top, width, height = box_info  # Assuming box_info is in the format (left_top, width, height)
            right_bottom = (left + width, top + height)
            left_top = (left, top)

        elif self.return_type == "v2":

            left, top, right, bottom = box_info  # Assuming box_info is in the format (left_top, width, height)
            left_top = (left, top)
            right_bottom = (right, bottom)

        elif self.return_type == "v3":

            center_x, center_y, width, height = box_info
            left_top = (center_x - width // 2, center_y - height // 2)
            right_bottom = (center_x + width // 2, center_y + height // 2)

        return left_top, right_bottom


    def vis_result_folder(self, left_top, right_bottom,img_path):
        """
        visualize images for folder
        left_top, right_bottom from convert_lefttop_rightbottom

        """
        img = cv2.imread(img_path)
        # Visualize the result on the image
        vis_img = img.copy()
        cv2.rectangle(vis_img, left_top, right_bottom, (0, 255, 255), 2)
        # Save the visualization image
        output_folder = os.path.dirname(img_path)
        output_dir = os.path.join(os.path.dirname(output_folder), f"vis_{args.return_type}_{args.net_type}")
        os.makedirs(output_dir, exist_ok=True)
        img_name = os.path.basename(img_path)
        output_path = os.path.join(output_dir, f"{img_name}")
        cv2.imwrite(output_path, vis_img)


    def vis_result_singleimage(self, left_top, right_bottom,img_path):
        """
        visualize images for singleimage
        left_top, right_bottom from convert_lefttop_rightbottom

        """
        img = cv2.imread(img_path)
        # Visualize the result on the image
        vis_img = img.copy()
        cv2.rectangle(vis_img, left_top, right_bottom, (0, 255, 255), 2)
        # Save the visualization image
        output_folder = os.path.dirname(img_path)  # 获取img_folder的上一级路径
        output_dir = os.path.join(os.path.dirname(output_folder), f"vis_{args.return_type}_{args.net_type}_singelimage")
        os.makedirs(output_dir, exist_ok=True)
        img_name = os.path.basename(img_path)
        output_path = os.path.join(output_dir, f"{args.return_type}_{img_name}")
        cv2.imwrite(output_path, vis_img)


    def detect_folder(self, img_folder):
        output_filename = f"output_{args.scale}_{args.net_type}_{args.return_type}.txt"  # 构造输出文件名
        output_folder = os.path.dirname(img_folder) 
        output_path = os.path.join(output_folder, output_filename)  # 构造输出文件路径
        with open(output_path, 'w') as f:
            for filename in os.listdir(img_folder):
                if filename.endswith(".jpg") or filename.endswith(".png"):
                    img_path = os.path.join(img_folder, filename)
                    relative_path = os.path.relpath(img_path, img_folder)  # 获取相对路径
                    box_info = self.detect_img(img_path)
                    f.write(f"{relative_path} {' '.join([str(i) for i in box_info])}\n")
                    left_top, right_bottom=self.convert_lefttop_rightbottom(img_path,box_info)
                    self.vis_result_folder(left_top, right_bottom,img_path)

        return box_info


    def process_images(self, img_paths):
        """
        not used in this py
        can return face box for the folder

        Args:
            img_paths: list, list of image paths
        Returns:
            results: list, list of box information according its return type for whole folder
        """
        results = []
        for img_path in img_paths:
            box_info = self.detect_img(img_path)
            results.append(box_info)
            print(results)
        return results


    def output_txt_from_return_type(self,results,root_dir):
        """
            output txt file in SLPT mode, txt format: img_path, box(according to return_type)
            Args:
                results: dict, boxes information
                root_dir: str, output dir
        """
        os.makedirs(root_dir,exist_ok=True)
        output_path_txt=os.path.join(root_dir,f"output_{args.scale}_{args.net_type}_{args.return_type}.txt")

        with open(output_path_txt, 'w') as f:
            for bbox in results:
                if bbox is None:
                    continue
                adjust_size = self.adjust_box_size_SLTP(bbox)
                box_txt_info = self.convert_to_return_type_SLPT(adjust_size)
                img_path = bbox['img_path']
                self.draw_box_for_SLPT(adjust_size, img_path)
                f.write(f"{img_path} {' '.join([str(i) for i in box_txt_info])}\n")
        
        
    def draw_box_for_SLPT(self,adjust_size,img_path):

        output_folder = os.path.join("vis_box", os.path.dirname(img_path))
        os.makedirs(output_folder, exist_ok=True)
        file_name = os.path.basename(img_path)
        output_path = os.path.join(output_folder, file_name)

        img=cv2.imread(img_path)
        vis_img=img.copy()
        cv2.rectangle(vis_img,(int(adjust_size[0]-adjust_size[2]/2),int(adjust_size[1]-adjust_size[2]/2)),(int(adjust_size[0]+adjust_size[2]/2),int(adjust_size[1]+adjust_size[2]/2)),(255, 0, 255), 2)
        cv2.imwrite(output_path, vis_img)
    
    def adjust_box_size_SLTP(self,bbox):
        
        """
            Adjust the bounding box size based on the scale, return a square box
            Args:
                bbox: dict, boxes information(left_top,h,w)
                self.scale
            Returns:
                adjust_size: list, boxes information(left_top,h,w)
        """

        edge_length = int(max(bbox['box'][2],bbox['box'][3])*self.scale)
        adjust_size = (int(bbox['box'][0]+bbox['box'][2]/2),int(bbox['box'][1]+bbox['box'][3]/2),edge_length,edge_length)
        return adjust_size
 
    def convert_to_return_type_SLPT(self,adjust_size):

        """
        get box information from return_type
            Args:
                adjust_size: list, boxes information(left_top,h,w)
                return_type: choice:[v1,v2,v3], v1 is return left-top point and wh,
                v2 is return left-top and right-bottom points, v3 is return center point and wh.
            Returns:
                3 type information
        """
        if self.return_type == "v1":
            return (int(adjust_size[0]-adjust_size[2]/2),int(adjust_size[1]-adjust_size[2]/2),adjust_size[2],adjust_size[3])

        elif self.return_type == "v2":
            return (int(adjust_size[0]-adjust_size[2]/2),int(adjust_size[1]-adjust_size[2]/2),int(adjust_size[0]+adjust_size[2]/2),int(adjust_size[1]+adjust_size[2]/2))
            
        else :
            return adjust_size

        





def parse_args():
    parser = argparse.ArgumentParser(description='Face Detecting')
    parser.add_argument('--net_type', default="mtcnn", type=str, choices=["mtcnn", "SLPT"], help='choose network')
    parser.add_argument('--return_type', default="v1", type=str, choices=["v1", "v2", "v3"],
                        help='choose return type')
    parser.add_argument('--scale', default=1.0, type=float, help='the scale of box size')
    parser.add_argument('--detect_folder', default=False, action='store_true', help="detect folder or img file")
    parser.add_argument('--detect_video', default=False, action='store_true', help="detect video")
    parser.add_argument('--path', required=True, type=str, help="img file path or folder path")
    parser.add_argument('--output_path', default="test_data_SLPT_folder", type=str, help="output root folder path")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    detector = FaceDetecter(args.net_type, args.return_type, args.scale)
    if detector.net_type == "mtcnn":

        if args.detect_folder:
            # MTCNN: detects and visualizes images from a single folder
            info=detector.detect_folder(args.path)
        else:
            # MTCCN: detects and visualizes single image
            info=detector.detect_img(args.path)
            left_top, right_bottom=detector.convert_lefttop_rightbottom(args.path, info)
            detector.vis_result_singleimage(left_top, right_bottom,args.path)
    
    elif detector.net_type == "SLPT":
        if args.detect_folder:
            # SLPT: detects and visualizes images from folders
            # ATTENTION: Visualized results and returns_type are independent
            results_correct = detector.net.detect_faces_from_folder_recur(args.path,warn_record=True)
            detector.output_txt_from_return_type(results_correct, "SLPT_txt")
            detector.net.vis_results_from_folder(args.output_path, results_correct)

        elif args.detect_video:
            # SLPT: detects and visualizes images from video
            detector.net.detect_vis_face_from_video(args.output_path, args.path)

        else:
            # SLPT: detects and visualizes single image 
            box_info = detector.net.detect_face_from_img_path(args.path, warn_record=True)
            detector.net.vis_result_from_img(args.output_path,box_info)
