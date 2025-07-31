import os
import cv2
import numpy as np
import logging
from keras_facenet import FaceNet

# Configure logging
LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
logging.basicConfig(
    level=logging.INFO,
    format=LOG_FORMAT,
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("face_recognizer.log", mode='a')
    ]
)
logger = logging.getLogger("FaceRecognizer")

def l2_normalize(x, axis=-1, epsilon=1e-10):
    output = x / np.sqrt(np.maximum(np.sum(np.square(x),
                         axis=axis, keepdims=True), epsilon))
    return output

def who_is_it(embeeds, database):
    min_dist = 100
    identity = "<UKN>"
    logger.debug("Starting identity search in database.")
    for (name, db_enc) in database.items():
        for enc in db_enc:
            dist = np.linalg.norm(embeeds - enc)
            logger.debug(f"Comparing with {name}, distance: {dist:.4f}")
            if dist < min_dist:
                min_dist = dist
                identity = name
    logger.info(f"Minimum distance found: {min_dist:.4f} for identity: {identity}")
    if min_dist > 1:
        identity = "<UKN>"
        logger.warning("No matching identity found. Returning <UKN>.")
    return identity

def load_dataset(model):
    database = {}
    dataset_path = 'dataset'
    logger.info(f"Loading dataset from {dataset_path}")
    if not os.path.exists(dataset_path):
        logger.error(f"Dataset path '{dataset_path}' does not exist.")
        return database
    for person in os.listdir(dataset_path):
        person_path = os.path.join(dataset_path, person)
        if not os.path.isdir(person_path):
            logger.warning(f"Skipping non-directory entry: {person_path}")
            continue
        database[person] = []
        logger.info(f"Processing images for person: {person}")
        for pic in os.listdir(person_path):
            img_path = os.path.join(person_path, pic)
            img = cv2.imread(img_path)
            if img is None:
                logger.warning(f"Failed to read image: {img_path}")
                continue
            try:
                embedding = model.extract(img, threshold=0.95)[0]['embedding']
                database[person].append(l2_normalize(embedding))
                logger.debug(f"Added embedding for {person}: {pic}")
            except Exception as e:
                logger.error(f"Error extracting embedding for {img_path}: {e}")
    logger.info("Finished loading dataset.")
    return database

# Constante
font = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 1
fontColor = (255, 255, 255)
lineType = 2
bottomLeftCornerOfText = (55, 25)

logger.info("Initializing FaceNet model.")
model = FaceNet()

logger.info("Creating the database of encoded faces.")
database = load_dataset(model)

cap = cv2.VideoCapture(0)
ret, frame = cap.read()
if not ret:
    logger.error("Failed to read from webcam.")
else:
    logger.info("Webcam stream started successfully.")

while ret:
    detections = model.extract(frame, threshold=0.95)
    logger.debug(f"Detections found: {len(detections)}")
    for detection in detections:
        x, y, h, w = detection['box']
        identity = who_is_it(l2_normalize(detection['embedding']), database)
        logger.info(f"Detected identity: {identity} at location: {detection['box']}")
        cv2.rectangle(frame, (x, y), (x+h, y+w), (0, 0, 255), 2)
        cv2.putText(frame, identity, (x-10, y-10), font,
                    fontScale, fontColor, lineType)

    cv2.imshow('img', frame)
    key = cv2.waitKey(10)
    if key == 27:
        logger.info("ESC pressed. Exiting.")
        break
    ret, frame = cap.read()
cv2.destroyAllWindows()
logger.info("Application terminated gracefully.")
