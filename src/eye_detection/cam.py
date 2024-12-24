import cv2
from PIL import Image
import numpy as np
import mediapipe as mp
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2

def replaceZeroes(data):
  min_nonzero = min(data[np.nonzero(data)])
  data[data == 0] = min_nonzero
  return data

def MSR(img, scales):
    weight = 1 / 3.0
    scales_size = len(scales)
    h, w = img.shape[:2]
    log_R = np.zeros((h, w), dtype=np.float32)

    for i in range(scales_size):
        img = replaceZeroes(img)
        L_blur = cv2.GaussianBlur(img, (scales[i], scales[i]), 0)
        L_blur = replaceZeroes(L_blur)
        dst_Img = cv2.log(img/255.0)
        dst_Lblur = cv2.log(L_blur/255.0)
        dst_Ixl = cv2.multiply(dst_Img, dst_Lblur)
        log_R += weight * cv2.subtract(dst_Img, dst_Ixl)

    dst_R = cv2.normalize(log_R,None, 0, 255, cv2.NORM_MINMAX)
    log_uint8 = cv2.convertScaleAbs(dst_R)
    return log_uint8
  

LEFT_EYE_LANDMARKS = [249, 263, 362, 373, 374, 380, 381, 382, 384, 385, 386, 387, 388, 390, 398, 466]
RIGHT_EYE_LANDMARKS = [7, 33, 133, 144, 145, 153, 154, 155, 157, 158, 159, 160, 161, 163, 173, 246]

# setting
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = FaceLandmarkerOptions(
  num_faces=10,
  base_options=BaseOptions(model_asset_path='src/eye_detection/model/face_landmarker.task'),
  running_mode=VisionRunningMode.IMAGE)

# create instance of FaceLandmarker  
landmarker = FaceLandmarker.create_from_options(options)

cap = cv2.VideoCapture(1)

while cap.isOpened():
  success, frame = cap.read() # read frame
  
  '''
  #----------------------------------test MSR(Retinex光照補強)-------------------------------------
  scales = [3, 5, 9]
  b_gray, g_gray, r_gray = cv2.split(frame)
  b_gray = MSR(b_gray, scales)
  g_gray = MSR(g_gray, scales)
  r_gray = MSR(r_gray, scales)
  result = cv2.merge([b_gray, g_gray, r_gray])
  cv2.imwrite("MSR_image.jpg", result)
  #-----------------------------------------------------------------------------------------------
  '''
  if not success:
    print('Can not get Frame')
    break
  
  H, W, C = frame.shape
  #frame = cv2.flip(frame, flipCode=1) #水平翻轉影像(no need)
  #rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
  rgb_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
  face_landmarker_result = landmarker.detect(rgb_image) # detect landmarks(need use mediapipe type image)
  
  face_landmarks_list = face_landmarker_result.face_landmarks
  
  #---------------TEST GET EYE FEATURE--------------------------
  left_eye = []
  left_eye_img = []
  flag = True
  for idx in range(len(face_landmarks_list)):
    face_landmarks = face_landmarks_list[idx]
    left_eye = [face_landmarks[i] for i in LEFT_EYE_LANDMARKS]
    left_eye_pts = [(int(pt.x * W), int(pt.y * H)) for pt in left_eye]
    left_x, left_y, left_w, left_h = cv2.boundingRect(np.array(left_eye_pts))
    #left_eye_img = frame[left_y:left_y + left_h, left_x:left_x + left_w]
    #cv2.rectangle(frame, (left_x, left_y), (left_x + left_w, left_y + left_h), (0, 255, 0), 2)
    
    #------------------Test fixed-size img---------------
    fixed_size_x = 39 #25
    fixed_size_y = 39 #10
    left_eye_mid = face_landmarks[473]
    left_eye_mid_pt_x = int(left_eye_mid.x*W)
    left_eye_mid_pt_y = int(left_eye_mid.y*H)
    #left_eye_mid_pt = [(int(pt.x * W), int(pt.y * H)) for pt in left_eye_mid]
    #!FIXME
    left_eye_img = frame[left_eye_mid_pt_y-fixed_size_y:left_eye_mid_pt_y+fixed_size_y+1, left_eye_mid_pt_x-fixed_size_x:left_eye_mid_pt_x+fixed_size_x+1]
          
    right_eye = [face_landmarks[i] for i in RIGHT_EYE_LANDMARKS]
    right_eye_pts = [(int(pt.x * W), int(pt.y * H)) for pt in right_eye]
    right_x, right_y, right_w, right_h = cv2.boundingRect(np.array(right_eye_pts))
    #cv2.rectangle(frame, (right_x, right_y), (right_x+right_w, right_y+right_h), (0, 255, 0), 2)
  if flag == True and len(left_eye_img) != 0:
    cv2.imwrite("left_eye.jpg", left_eye_img)
    flag = False
  '''
  # Loop through the detected faces to visualize.
  for idx in range(len(face_landmarks_list)):
    face_landmarks = face_landmarks_list[idx]

    # get landmark use 
    face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    face_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in face_landmarks
    ])  
  
  
    connections = frozenset().union(*[
      mp.solutions.face_mesh.FACEMESH_LEFT_EYE,
      mp.solutions.face_mesh.FACEMESH_RIGHT_EYE
    ])
    solutions.drawing_utils.draw_landmarks(
      image=frame,
      landmark_list=face_landmarks_proto,
      connections=connections,
      landmark_drawing_spec=None,
      connection_drawing_spec=mp.solutions.drawing_styles
      .get_default_face_mesh_contours_style())
  '''
  
  #-------------------------cv2 image process----------------------------
  if len(left_eye_img)!=0: #判斷是否有偵測到眼睛
    #!FIXME
    left_eye_img_o = cv2.imread('left_eye.jpg')
    #left_eye_img = Image.fromarray(left_eye_img)
    left_eye_img_gray = cv2.cvtColor(left_eye_img_o, cv2.COLOR_BGR2GRAY)
    left_eye_img_gray_blur = cv2.GaussianBlur(left_eye_img_gray, (3,3), 0)
    threshold, left_eye_img_otsu  = cv2.threshold(
      left_eye_img_gray_blur,
      0,
      255,
      cv2.THRESH_BINARY+cv2.THRESH_OTSU,
    )
    cv2.imwrite('left_eye_otsu.jpg', left_eye_img_otsu)
  #-----------------------------------------------------------------------
  
  
  # get frame after do landmark
  cv2.imshow("Face landmarks", frame)
  if cv2.waitKey(1) == 27: # press ESC to leave
    break
  
cap.release()
cv2.destroyAllWindows()