import cv2
import numpy as np
import os
import streamlit as st
from PIL import Image
import pyttsx3
import threading

# Load YOLO model
if not os.path.exists("yolov3.weights") or not os.path.exists("yolov3.cfg") or not os.path.exists("coco.names"):
    st.error("Error: YOLO model files not found!")
    st.stop()

net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

def detect_objects(image):
    blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(output_layers)

    boxes, confidences, class_ids = [], [], []
    height, width = image.shape[:2]
    
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x, center_y = int(detection[0] * width), int(detection[1] * height)
                w, h = int(detection[2] * width), int(detection[3] * height)
                x, y = int(center_x - w / 2), int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    return boxes, confidences, class_ids, indices

def draw_labels(image, boxes, confidences, class_ids, indices):
    detected_objects = []
    if len(indices) > 0:
        for i in indices.flatten():
            x, y, w, h = boxes[i]
            label = f"{classes[class_ids[i]]}: {confidences[i]:.2f}"
            detected_objects.append(classes[class_ids[i]])
            color = colors[class_ids[i]]
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return image, detected_objects

def speak_detected_objects(objects):
    if len(objects) == 1:  
        def speak():
            engine = pyttsx3.init()
            text = f"Detected object: {objects[0]}"
            engine.say(text)
            engine.runAndWait()
        threading.Thread(target=speak).start()

# Streamlit UI Configuration
st.set_page_config(page_title="YOLO Object Detection", page_icon="🔍", layout="wide")

# Custom CSS for Sky Blue Theme
st.markdown("""
    <style>
        body { background-color: #E3F2FD; color: black; }
        .stApp { background-color: #E3F2FD; }
        .css-18e3th9 { background-color: #90CAF9; }
        .stMarkdown h1 { text-align: center; color: #0D47A1; }
        .stSidebar { background-color: #64B5F6; color: white; }
        .stButton > button { width: 100%; background-color: #1976D2; color: white; }
        .stFileUploader { border: 2px solid #1976D2; }
    </style>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.title("Upload Image or Start Camera")
choice = st.sidebar.radio("Choose an option:", ["Upload Image", "Live Camera"], index=0)

st.markdown("<h1>Real-Time Object Detection</h1>", unsafe_allow_html=True)

if choice == "Upload Image":
    uploaded_file = st.sidebar.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file)
        image = np.array(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        boxes, confidences, class_ids, indices = detect_objects(image)
        detected_image, detected_objects = draw_labels(image, boxes, confidences, class_ids, indices)
        
        st.image(cv2.cvtColor(detected_image, cv2.COLOR_BGR2RGB), caption="Detected Objects", use_column_width=True)
        speak_detected_objects(detected_objects)

elif choice == "Live Camera":
    cap = cv2.VideoCapture(0)
    stop = st.sidebar.button("Stop Camera")
    stframe = st.empty()
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or stop:
            break
        boxes, confidences, class_ids, indices = detect_objects(frame)
        detected_frame, _ = draw_labels(frame, boxes, confidences, class_ids, indices)
        stframe.image(cv2.cvtColor(detected_frame, cv2.COLOR_BGR2RGB), channels="RGB")
    
    cap.release()
    cv2.destroyAllWindows()
