# - cài đặt thư viện Streamlit: pip install streamlit
# - cài đặt thư viện Aspose Imaging: pip install aspose-imaging-python-net
# - clone github detectron2 và cài đặt framework Detectron2:
#       git clone https://github.com/facebookresearch/detectron2.git
#       python -m pip install -e detectron2
# - chỉnh sửa đường dẫn đến file cấu hình gốc của model trong framework Detectron2 (trong github detectron2)
# - chạy chương trình: python -m streamlit run demo.py

import numpy as np
import os
import cv2
import streamlit as st
import pydicom
import tempfile
import zipfile
import torch

from aspose.imaging import Image
from aspose.imaging.imageoptions import JpegOptions
from ultralytics import YOLO

from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor

NUM_CLASSES = 2
MAX_KIDNEYS = 2
THRESHOLD_RETINANET = 0.5
THRESHOLD_FASTERRCNN = 0.5
THRESHOLD_YOLODETECT = 0.5
THRESHOLD_YOLOSEGMENT = 0.5
BOX_THICKNESS = 1
BOX_COLOR = (255, 0, 0)
GPU = 0 # 0 nếu có 1 GPU, [0, 1] nếu có 2,...
DEFAULT_DX = 0.5
DEFAULT_DY = 0.5

# file cấu hình gốc
INIT_CFG_FASTERRCNN_PATH = '../detectron2/configs/COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml'
INIT_CFG_RETINANET_PATH = '../detectron2/configs/COCO-Detection/retinanet_R_101_FPN_3x.yaml'

MODEL_DETECT_RETINANET_PATH = './model/retinanet_kidneyDetect.pth'
MODEL_DETECT_FASTERRCNN_PATH = './model/fasterrcnn_kidneyDetect.pth'
MODEL_DETECT_YOLO_PATH = './model/yolo11x_kidneyDetect.pt'
MODEL_SEGMENT_YOLO_PATH = './model/yolo11x_stonesSegment.pt'
    
def loadModel_RetinaNet():
    cfg = get_cfg()
    cfg.merge_from_file(INIT_CFG_RETINANET_PATH)
    cfg.MODEL.WEIGHTS = MODEL_DETECT_RETINANET_PATH
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = THRESHOLD_RETINANET
    cfg.MODEL.RETINANET.NUM_CLASSES = NUM_CLASSES
    predictor = DefaultPredictor(cfg)
    return predictor

def loadModel_FasterRCNN():
    cfg = get_cfg()
    cfg.merge_from_file(INIT_CFG_FASTERRCNN_PATH)
    cfg.MODEL.WEIGHTS = MODEL_DETECT_FASTERRCNN_PATH
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = THRESHOLD_FASTERRCNN
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = NUM_CLASSES
    predictor = DefaultPredictor(cfg)
    return predictor

def readDICOM(dcmFile):
    # Đọc file DICOM
    dicom_data = pydicom.dcmread(dcmFile)
    dX, dY = None, None
    if 'PixelSpacing' in dicom_data:
        pixel_spacing = dicom_data.PixelSpacing
        dX = float(pixel_spacing[0])
        dY = float(pixel_spacing[1])
        st.write(f"Pixel spacing: dX = {dX}, dY = {dY}")
                        
    # Tạo và thiết lập đối tượng của lớp ImageOptionsBase, ở đây là JpegOptions
    jpeg_options = JpegOptions()
    image = Image.load(dcmFile)
    image.save('temp.jpg', jpeg_options)
    img = cv2.imread('temp.jpg')
    return img, dX, dY

def draw_rec(img, x_min, y_min, x_max, y_max, thickness):
    draw_img = img
    color = BOX_COLOR
    
    cv2.rectangle(draw_img, (x_min, y_min), (x_max, y_max), color, thickness)
    return draw_img

def count_pixels(polygon, img_shape):
    polygon_int = np.round(polygon).astype(np.int32)

    # Create a blank image
    img_shape = (img_shape[0], img_shape[1])
    blank_image = np.zeros(img_shape, dtype=np.uint8)

    # Draw the polygon
    cv2.fillPoly(blank_image, [polygon_int], color=255)

    # Count non-zero pixels (area of the polygon in pixels)
    pixel_count = np.count_nonzero(blank_image)
    return pixel_count

def segment_and_calculate_areaOfStone(img, coordinates, dX, dY, fileName, tmp_dir, image_paths):
    pixel_size = dX*dY
    kidneyIndex = 1
    for coordinate in coordinates:
        col1, col2 = st.columns(2)
        x_min, y_min, x_max, y_max = coordinate
        kidney_image = img[y_min:y_max, x_min:x_max]
        kidneyImagePath = f"{fileName}_kidney{kidneyIndex}.png"
        kidneyImageName, _ = os.path.splitext(kidneyImagePath)
                        
        # Segment stones
        if torch.cuda.is_available():
            segmentResults = modelSegment_YOLO(kidney_image, conf=THRESHOLD_YOLOSEGMENT, device=0)
        else:
            segmentResults = modelSegment_YOLO(kidney_image, conf=THRESHOLD_YOLOSEGMENT)
            
        masks = segmentResults[0].masks
        boxes = segmentResults[0].boxes
                    
        with col2:
            if masks is not None and hasattr(masks, 'xy') and boxes is not None and hasattr(boxes, 'xyxy'):
                xy_coords = masks.xy
                xyxy_coords = boxes.xyxy
                stoneIndex = 1
                stone_image = kidney_image.copy()
                            
                for i, (coords, box_coords) in enumerate(zip(xy_coords, xyxy_coords), start=1):                                
                    num_coords = count_pixels(coords, stone_image.shape)
                    size_ofStone = num_coords*pixel_size
                    
                    xm, ym, xM, yM = box_coords.tolist()
                    kidneyStone_image = stone_image[round(ym):round(yM), round(xm):round(xM)]
                                
                    kidney_image = draw_rec(kidney_image, round(xm), round(ym), round(xM), round(yM), BOX_THICKNESS)
                    kidneyStoneImagePath = f"{num_coords}_{size_ofStone}_{kidneyImageName}_stone{stoneIndex}.png"
                                
                    st.image(kidneyStone_image)
                    st.write(f'{num_coords} pixels - {size_ofStone} square milimetre')
                    stoneIndex += 1
                                
                    image_path = os.path.join(tmp_dir, kidneyStoneImagePath)
                    cv2.imwrite(image_path, kidneyStone_image)
                    image_paths.append(image_path)
                            
            else: st.write('This kidney is healthy')
                        
        with col1:
            st.image(kidney_image, caption=f"Kidney {kidneyIndex}")
            kidneyIndex += 1
        st.divider()

def segmentStones_and_saveToZip(modelDetect_option, imageType=True):
    # Initialize session state
    if "uploaded_files" not in st.session_state:
        st.session_state.uploaded_files = []
    if "uploader_key" not in st.session_state:
        st.session_state.uploader_key = 0  # Dynamic key for the file uploader

    if imageType:
        label_text = "Choose images..."
        fileType = ["png", "jpg", "jpeg"]
    else:
        label_text = "Choose dicom files..."
        fileType = ["dcm"]
    
    # File uploader
    uploaded_files = st.file_uploader(
        label=label_text, 
        accept_multiple_files=True, 
        type=fileType,
        key=f"uploader_{st.session_state.uploader_key}"
    )

    # Store uploaded files in session state
    if uploaded_files:
        st.session_state.uploaded_files = uploaded_files
    
    if st.session_state.uploaded_files:
        with tempfile.TemporaryDirectory() as tmp_dir:
            image_paths = []
            
            seen_filenames = set()
            unique_uploaded_files = []

            for uploaded_file in uploaded_files:
                if uploaded_file.name not in seen_filenames:
                    seen_filenames.add(uploaded_file.name)
                    unique_uploaded_files.append(uploaded_file)

            uploaded_files = unique_uploaded_files
            
            for i, uploaded_file in enumerate(uploaded_files):
                filePath = uploaded_file.name
                fileName, _ = os.path.splitext(filePath)
                
                if imageType:
                    dX_col, dY_col = st.columns(2)
                    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                    img_cvt = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    st.write('Insert Pixel spacing (mm)')
                    with dX_col:
                        dX = st.number_input(
                            'Insert Pixel spacing (mm) - dX (spacing between the centers of adjacent rows)',
                            value=DEFAULT_DX,
                            format='%0.6f',
                            key=f"dX{fileName}"
                        )
                    with dY_col:
                        dY = st.number_input(
                            'Insert Pixel spacing (mm) - dY (spacing between the centers of adjacent columns)',
                            value=DEFAULT_DY,
                            format='%0.6f',
                            key=f"dY{fileName}"
                        )
                else:
                    img, dX, dY = readDICOM(uploaded_file)
                    img_cvt =  cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    
                    if not (dX and dY):
                        st.write('Can not read Pixel spacing in dicom file.')
                        dX_col, dY_col = st.columns(2)
                        with dX_col:
                            dX = st.number_input(
                                'Insert Pixel spacing (mm) - dX (distance between the centers of two adjacent pixels horizontally)',
                                value=0.92578,
                                format='%0.6f',
                                key=f"dX{fileName}"
                            )
                        with dY_col:
                            dY = st.number_input(
                                'Insert Pixel spacing (mm) - dY (distance between the centers of two adjacent pixels vertically)',
                                value=0.92578,
                                format='%0.6f',
                                key=f"dY{fileName}"
                            )
                                            
                st.image(img_cvt, caption=fileName, use_container_width=True)
                coordinates = []
                if modelDetect_option == 'YOLO11':
                    if torch.cuda.is_available():
                        results = modelDetect_YOLO(img_cvt, conf=THRESHOLD_YOLODETECT, max_det=MAX_KIDNEYS, device=GPU)
                    else:
                        results = modelDetect_YOLO(img_cvt, conf=THRESHOLD_YOLODETECT, max_det=MAX_KIDNEYS)
                        
                    if len(results[0].boxes.xyxy) <= 0:
                        st.write('Can not detect kidney in ', fileName)
                        continue
                    
                    for coordinate in results[0].boxes.xyxy:
                        x_min, y_min, x_max, y_max = coordinate.tolist()
                        coordinates.append((round(x_min), round(y_min), round(x_max), round(y_max)))
                
                else:
                    if modelDetect_option == 'Faster R-CNN':
                        outputs = modelDetect_FasterRCNN(img_cvt)
                    else:
                        outputs = modelDetect_RetinaNet(img_cvt)
                    
                    instances = outputs["instances"]
                    boxes = instances.pred_boxes.tensor.cpu().numpy()
                    scores = instances.scores.cpu().numpy()
                    if not scores.any():
                        st.write('Can not detect kidney in ', fileName)
                        continue
                    
                    # Sắp xếp theo điểm số giảm dần và chọn tối đa 2 bbox
                    sorted_indices = np.argsort(scores)[::-1][:2]
                    top_boxes = boxes[sorted_indices]
                    
                    for box in top_boxes:
                        box = np.round(box).astype(int)
                        coordinates.append((box[0], box[1], box[2], box[3]))
                    
                segment_and_calculate_areaOfStone(img, coordinates, dX, dY, fileName, tmp_dir, image_paths)
            
            zip_path = os.path.join(tmp_dir, "stoneInKidney_images.zip")
            with zipfile.ZipFile(zip_path, 'w') as zipf:
                for image_path in image_paths:
                    zipf.write(image_path, os.path.basename(image_path))
                    
            with open(zip_path, "rb") as f:
                button_col1, button_col2 = st.columns(2)
                with button_col1:
                    if st.button('Delete all'):
                        # Reset uploaded files in session state after download
                        st.session_state.uploaded_files = []
                        st.session_state.uploader_key += 1  # Update uploader key to reset widget
                        st.rerun()  # Reload the app to reflect changes
                
                with button_col2:    
                    if st.download_button(
                        label="Download all stone images as ZIP",
                        data=f,
                        file_name="stoneInKidney_images.zip",
                        mime="application/zip"
                    ):
                        # Reset uploaded files in session state after download
                        st.session_state.uploaded_files = []
                        st.session_state.uploader_key += 1  # Update uploader key to reset widget
                        st.rerun()  # Reload the app to reflect changes

if __name__ == "__main__":
    st.set_page_config(page_title="Phát hiện - Phân đoạn sỏi thận")
    st.markdown(
        "<h1 style='text-align: center; color: black;'>Phát hiện & Phân đoạn sỏi thận</h1>", 
        unsafe_allow_html=True,
    )
    
    # Load YOLO model
    modelDetect_YOLO = YOLO(MODEL_DETECT_YOLO_PATH)
    modelSegment_YOLO = YOLO(MODEL_SEGMENT_YOLO_PATH)
    
    # Load Faster R-CNN model
    modelDetect_FasterRCNN = loadModel_FasterRCNN()
    
    # Load DEtection TRansformer model
    modelDetect_RetinaNet = loadModel_RetinaNet()
    
    # modelDetect_option = st.selectbox(
    #     'Which model do you use for detect kidneys?',
    #     ('YOLO11', 'Faster R-CNN', 'RetinaNet')
    # )
    modelDetect_option = 'Faster R-CNN'
    
    filesType_option = st.selectbox(
        'What type of files are you going to use?',
        ('image/PNG - image/JPG - image/JPEG', 'dicom/DCM')
    )
    
    if filesType_option == 'image/PNG - image/JPG - image/JPEG':
        segmentStones_and_saveToZip(modelDetect_option)
    else: segmentStones_and_saveToZip(modelDetect_option, imageType=False)    