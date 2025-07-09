from tensorflow.keras.applications.inception_v3 import InceptionV3

def load_model():
    model = InceptionV3(weights='imagenet')
    return model
