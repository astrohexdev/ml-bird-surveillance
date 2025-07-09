import cv2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import decode_predictions
from model.incept_model import load_model
from utils.preprocess import preprocess

model = load_model()

def predict(frame):
    resized_frame = cv2.resize(frame, (299, 299))
    img = image.array_to_img(resized_frame)
    preprocessed_img = preprocess(img)

    predictions = model.predict(preprocessed_img)
    decoded_predictions = decode_predictions(predictions, top=1)[0]
    top_prediction = decoded_predictions[0]
    
    label = f"{top_prediction[1]} ({top_prediction[2]:.2f})"
    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)
