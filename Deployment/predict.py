import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

class CustomModel(tf.keras.Model):
    def __init__(self):
        super(CustomModel, self).__init__()
        # Define your model layers here
        self.batch_normalization = tf.keras.layers.BatchNormalization()

    def call(self, inputs):
        # Define the forward pass of your model
        # Example:
        x = self.batch_normalization(inputs)
        # Add more layers as needed
        return x

def run():
    st.subheader('Vegetables Prediction')
    st.write('Note : You can only predict : apple, banana, cabbage, eggplant, garlic, jalepeno, kiwi, lemon, mango, onion, orange, paprika, pear, raddish, soy beans, tomato, watermelon')
    st.write('')
    
    # Load the model
    model = load_model('best_model.h5', custom_objects={'CustomModel': CustomModel})

    uploaded_file = st.file_uploader('Upload an image', type=['png', 'jpg', 'jpeg'])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Preprocess the image
        resized_image = image.resize((110, 110))  # Resize the image to match the input size of the model
        normalized_image = np.array(resized_image) / 255.0  # Normalize the pixel values
        input_image = np.expand_dims(normalized_image, axis=0)  # Add an extra dimension as the model expects a batch of images

        # Perform image classification
        prediction = model.predict(input_image)
        predicted_class = np.argmax(prediction)

        class_names = ['apple', 'banana', 'cabbage', 'eggplant', 'garlic', 'jalepeno', 'kiwi', 'lemon', 'mango', 'onion', 'orange', 'paprika', 'pear', 'raddish', 'soy beans', 'tomato', 'watermelon']
        # Display the predicted class label
        predicted_label = class_names[predicted_class]
        st.write('Predicted Class:', predicted_label)

if __name__=='__main__':
    run()
