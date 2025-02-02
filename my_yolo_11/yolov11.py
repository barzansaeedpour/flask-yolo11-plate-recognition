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

classNames = ['plate']
# classNames2 = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'be', 'dal', 'ein', 'ghaf', 'h', 'jim', 'lam', 'mim', 'noon', 'sad', 'sin', 'ta', 'te', 'waw', 'ye']
numbers = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',]
# letters = ['alef', 'be', 'che', 'dal', 'ein', 'ghaf', 'h', 'jim', 'lam', 'mim', 'noon', 'sad', 'sin', 'ta', 'te', 'waw', 'ye']
# letters = ['alef', 'be', 'che', 'ein', 'dal',  'ghaf', 'h', 'jim', 'lam', 'mim', 'noon', 'sad', 'sin', 'ta', 'te', 'waw', 'ye']
# letters = ['alef', 'be', 'che', 'dal', 'ein', 'jim', 'h', 'ghaf', 'lam', 'sad', 'noon', 'mim', 'sin', 'ta', 'te', 'waw', 'ye']
letters = ['be', 'dal', 'ein', 'ghaf', 'h', 'jim', 'lam', 'mim', 'noon', 'sad', 'sin', 'ta', 'te', 'waw', 'ye']


classNames2 = numbers + letters
# classNames2 = list(model_character_detection.names.values())


def plate_detection(frame, model_plate_detection, model_character_detection, save_dir,save = True):
    img = frame
    results = model_plate_detection.predict(source=img, conf = 0.3, save=False, show = False, project=save_dir, name="", save_txt = False) 
    for r in results:
            boxes = r.boxes

    for box in boxes:
        # bounding box
        x1, y1, x2, y2 = box.xyxy[0]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # convert to int values

        try:
            extracted_plate_image = extract_the_plate(img=img, top_left=(x1, y1), bottom_right=(x2, y2))
        except:
            continue
        new_name = get_new_name()
        
        ########################################### Start Rotate image
        # (h, w) = extracted_plate_image.shape[:2]

        # # Calculate the center of the image
        # center = (w // 2, h // 2)

        # # Define the rotation matrix
        # M = cv2.getRotationMatrix2D(center, -15, 1.0)

        # # Perform the rotation
        # rotated = cv2.warpAffine(extracted_plate_image, M, (w, h))
        # # Save or display the rotated image
        # cv2.imwrite('textracted_plate_image.jpg', extracted_plate_image)
        # cv2.imwrite('trotated.jpg',rotated)
        # extracted_plate_image = rotated
        ########################################### end Rotate image
        
        
        results2 = model_character_detection.predict(source=extracted_plate_image, conf = 0.7, save=False, show = False, project=save_dir, name="", save_txt = False) 
        
    
        for r in results2:
            boxes = r.boxes
            detected_classes = [int(box.cls) for box in boxes]   
            
            detected_classes = [classNames2[i] for i in detected_classes]
            continue_flag, detected_classes = check_detected_classes_validation(detected_classes, numbers, letters, save_dir, boxes)
        # if continue_flag:
        #     continue    
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
           
            cv2.imwrite(save_dir + new_name +'-detected.png',img)
        time.sleep(0.1)
        
        return detected_classes, frame, img
        # cv2.imshow("Real-time Webcam", img)
        # time.sleep(0.1)


    return '', [], []