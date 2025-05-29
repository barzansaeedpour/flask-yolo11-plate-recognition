from ultralytics import YOLO
import cv2
import os
import shutil
import numpy as np
import math
import time
from PIL import ImageFont, ImageDraw, Image
# from my_yolo_11.rabbitmq.publisher import publish
from dotenv import find_dotenv, load_dotenv
from my_yolo_11.utils.utils import extract_the_plate, check_detected_classes_validation, get_new_name, persian
import joblib
import torch

dotenv_path = find_dotenv()
load_dotenv(dotenv_path)
base_dir = os.getenv("base_dir_plate_detection")

# # model = YOLO(model_path)
# # joblib.dump(model, cache_path)
# # print("Model loaded from file and cached.")

# model_plate_detection_path = f"{base_dir}/my_yolo_v8/models/plate-detector.pt"
# model_plate_detection_cache_path = f"{base_dir}/my_yolo_v8/models/plate-detector-cache.pt"
# model_character_detection_path = f"{base_dir}/my_yolo_v8/models/character-detector.pt"
# model_character_detection_cache_path = f"{base_dir}/my_yolo_v8/models/character-detector-cache.pt"
# # model_plate_detection = YOLO(cache_path)
# # joblib.dump(model_plate_detection, cache_path)

# try:
#     # Try to load the model from the cache
#     model_plate_detection = torch.load(model_plate_detection_cache_path)
#     model_character_detection = torch.load(model_character_detection_cache_path)
#     print("Model loaded from cache.")
# except FileNotFoundError:
#     # If cache doesn't exist, load the model and save it to the cache
#     model_plate_detection = YOLO(model_plate_detection_path)
#     model_character_detection = YOLO(model_character_detection_path)
#     torch.save(model_plate_detection, model_plate_detection_cache_path)
#     torch.save(model_character_detection, model_character_detection_cache_path)
#     print("Model loaded from file and cached.")


# model_character_detection = YOLO()


# model_plate_detection.predict(source=np.zeros([640,640,3]), conf = 0.1, save=False, show = False,  name="", save_txt = False)
# model_character_detection.predict(source=np.zeros([640,640,3]), conf = 0.1, save=False, show = False, name="", save_txt = False)

# plate_detection_output_path = "/code/my_yolo_v8/outputs/"
# plate_detection_output_path = ""
# try:
#     shutil.rmtree(plate_detection_output_path)
# except:
#     pass
# os.makedirs(plate_detection_output_path, exist_ok=True)

# plate_detection_path = f'{base_dir}/my_yolo_v8/outputs2/plate_detection_path/'
# character_detection_path = f'{base_dir}/my_yolo_v8/outputs2/character_detection_path/'

import cv2
import numpy as np

def add_filled_rectangle(image, position, text, color=(255, 255, 255), alpha=0.9, padding=10):
    x, y = map(int, position)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 2
    thickness = 2

    # Calculate text size
    (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
    box_width = text_width + 2 * padding
    box_height = text_height + 2 * padding

    # Define rectangle corners
    top_left = (x, y - box_height - 10)
    bottom_right = (x + box_width, y - 10)

    # Create overlay for transparency
    overlay = image.copy()
    cv2.rectangle(overlay, top_left, bottom_right, color, thickness=-1)

    # Blend with transparency
    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)

    return image, top_left, text_height, padding


def add_text_to_image(image, text, position):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 2
    color = (0, 0, 0)
    thickness = 2

    # Add transparent white rectangle first
    image, top_left, text_height, padding = add_filled_rectangle(image, position, text)

    # Draw text centered vertically inside the rectangle
    text_x = top_left[0] + padding
    text_y = top_left[1] + text_height + padding - 2

    cv2.putText(image, text, (text_x, text_y), font, font_scale, color, thickness, lineType=cv2.LINE_AA)
    return image



classNames = ['plate']
# classNames2 = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'be', 'dal', 'ein', 'ghaf', 'h', 'jim', 'lam', 'mim', 'noon', 'sad', 'sin', 'ta', 'te', 'waw', 'ye']
numbers = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',]
# letters = ['alef', 'be', 'che', 'dal', 'ein', 'ghaf', 'h', 'jim', 'lam', 'mim', 'noon', 'sad', 'sin', 'ta', 'te', 'waw', 'ye']
# letters = ['alef', 'be', 'che', 'ein', 'dal',  'ghaf', 'h', 'jim', 'lam', 'mim', 'noon', 'sad', 'sin', 'ta', 'te', 'waw', 'ye']
# letters = ['alef', 'be', 'che', 'dal', 'ein', 'jim', 'h', 'ghaf', 'lam', 'sad', 'noon', 'mim', 'sin', 'ta', 'te', 'waw', 'ye']
# letters = ['be', 'dal', 'ein', 'ghaf', 'h', 'jim', 'lam', 'mim', 'noon', 'sad', 'sin', 'ta', 'te', 'waw', 'ye']
letters = ['alef','be','pe','te','se','jim','dal','sin','sad','ta','ein','fe','ghaf','lam','mim','noon','waw','he','ye',]


classNames2 = numbers + letters
# classNames2 = list(model_character_detection.names.values())


def plate_detection(frame, model_plate_detection, model_character_detection, save_dir,save = True):
    
    
    ########################################### Keypoint Prediction
    img = frame

    model_path = "./my_yolo_11/models/best_pose_detection.pt"
    model = YOLO(model_path)
    
    image = np.ascontiguousarray(img)  # Make sure input is contiguous
    cv2.imwrite('./image.png', image)
    results = model.predict(source=image, conf=0.1, save=False,
                            show=False, project=save_dir, name="", save_txt=False)
    
    list_of_detected_plate_positions = []
    for result in results:
        
        # # Get the annotated image with keypoints and skeletons
        # annotated_image = result.plot()  # This returns an image with keypoints drawn
        # annotated_image = frame
        
        annotated_image = result.orig_img.copy()

        boxes = result.boxes.xyxy.cpu().numpy()      # shape: [N, 4]
        keypoints = result.keypoints.xy.cpu().numpy()  # shape: [N, K, 2]

        for i in range(len(boxes)):
            x1, y1, x2, y2 = map(int, boxes[i])
            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)

            # Draw keypoints
            for (x, y) in keypoints[i]:
                if x > 0 and y > 0:  # skip invisible points
                    cv2.circle(annotated_image, (int(x), int(y)), radius=4, color=(0, 0, 255), thickness=-1)

        
        
        # Convert BGR (OpenCV) to RGB (Matplotlib)
        # annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
        keypoints = result.keypoints.xy.cpu().numpy()  # Get keypoints in numpy format

        try:
            # image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            image_rgb = img 
            pts_src = np.array(keypoints[0], dtype=np.float32)
            if pts_src is not None:
                list_of_detected_plate_positions.append(pts_src[0])
            # Compute width and height of the new cropped region
            width = max(np.linalg.norm(
                pts_src[0] - pts_src[1]), np.linalg.norm(pts_src[2] - pts_src[3]))
            height = max(np.linalg.norm(
                pts_src[0] - pts_src[3]), np.linalg.norm(pts_src[1] - pts_src[2]))
            width, height = int(width), int(height)
            # Define the destination points to map the cropped region to a rectangle
            pts_dst = np.array([
                [0, 0],
                [width, 0],
                [width, height],
                [0, height]
            ], dtype=np.float32)
            # Compute the perspective transformation matrix
            matrix = cv2.getPerspectiveTransform(pts_src, pts_dst)
            # Apply the transformation to get the cropped region
            cropped = cv2.warpPerspective(image_rgb, matrix, (width, height))
            extracted_plate_image = cropped
            extracted_plate_image = cv2.resize(
                extracted_plate_image, (640, 360))
            # if extracted_plate_image is not None:
            #     extracted_plate_image = cv2.resize(extracted_plate_image, (640, 360))  # تغییر سایز تصویر تشخیص کاراکتر
            #     cv2.imshow('detected_plate Image', extracted_plate_image)
            # # امکان خروج از نمایش با فشردن کلید 'q'
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break
        except:
            continue
        frame_with_plate = np.array(annotated_image)
        
            
        
        ########################################### end Rotate image
        
        cv2.imwrite('./extracted_plate_image.png', extracted_plate_image)
        results2 = model_character_detection.predict(source=extracted_plate_image, conf = 0.3, save=False, show = False, project=save_dir, name="", save_txt = False) 
        
    
        for r in results2:
            boxes = r.boxes
            detected_classes = [int(box.cls) for box in boxes]   
            
            detected_classes = [classNames2[i] for i in detected_classes]
            continue_flag, detected_classes = check_detected_classes_validation(detected_classes, numbers, letters, save_dir, boxes)
        if continue_flag:
            continue    
        img = Image.fromarray(extracted_plate_image)
        for box in boxes:
            # bounding box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # convert to int values

            # confidence
            confidence = math.ceil((box.conf[0]*100))/100
            print("Confidence --->", confidence)

            # class name
            cls = int(box.cls[0])
            print("Class name -->", classNames2[cls])

            # object details
            org = [x1, y1-20]
            
            try:
                img = Image.fromarray(img)
            except:
                pass
            draw = ImageDraw.Draw(img)

            text = classNames2[cls]
            
            if confidence<0.80:
                color = (0, 0, 255)  # Red color
            else:
                color = (255,100,100)
            
            font = ImageFont.truetype(f"{base_dir}/my_yolo_v8/fonts/arial.ttf", 12)    
            
            draw.rectangle([(x1, y1), (x2, y2)], outline =color,)
            # draw.rectangle([(org[0], org[1]), (org[0]+(len(text)*25), org[1]+25)], fill =color)
            draw.rectangle([(org[0], org[1]), (org[0]+65, org[1]+25)], fill =color)
            draw.text(org, f"{persian(text)} -> %{round(confidence*100,2)}", font=font,fill=(255,255,255))
            img = np.array(img)

        img = np.array(img)
        
        if save:
            new_name = get_new_name()
            cv2.imwrite(save_dir + new_name +'-detected.png',img)
        time.sleep(0.1)
        frame_with_plate = add_text_to_image(frame_with_plate, detected_classes, list_of_detected_plate_positions[0])
        return detected_classes, frame_with_plate, img
        # cv2.imshow("Real-time Webcam", img)
        # time.sleep(0.1)


    return '', [], []