import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

# Load the trained models
cnn_model = tf.keras.models.load_model('C://Users//HP//Desktop//Final_train-models//cnn_model_92.h5')
densenet_model = tf.keras.models.load_model('C://Users//HP//Desktop//Final_train-models//densenet_model_88.h5')
inceptionv3_model = tf.keras.models.load_model('C://Users//HP//Desktop//Final_train-models//inceptionv3_model_73.h5')

# Define the image dimensions
img_width, img_height = 224, 224

# Define the class names
class_names = ['Mild', 'Moderate', 'No_DR', 'Proliferate_DR', 'Severe']

# Define the function to preprocess the image
def preprocess(image):
    image = image.resize((img_width, img_height))
    image = np.array(image)
    image = np.reshape(image, (1, img_width, img_height, 3))
    image = image / 255.
    return image

# Define the Streamlit app
def app():
    st.title('Diabetic Retinopathy Detection')

    uploaded_file = st.file_uploader('Choose an image', type=['jpg', 'jpeg', 'png'])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        image = preprocess(image)

        with tf.device('/CPU:0'):  # Specify the device for inference (CPU)
            cnn_prediction = cnn_model.predict(image)
            cnn_predicted_class = class_names[np.argmax(cnn_prediction)]
            cnn_probability = np.max(cnn_prediction)

            densenet_prediction = densenet_model.predict(image)
            densenet_predicted_class = class_names[np.argmax(densenet_prediction)]
            densenet_probability = np.max(densenet_prediction)

            inceptionv3_prediction = inceptionv3_model.predict(image)
            inceptionv3_predicted_class = class_names[np.argmax(inceptionv3_prediction)]
            inceptionv3_probability = np.max(inceptionv3_prediction)

        data = {
            'Model': ['CNN', 'densenet', 'inceptionv3'],
            'Predicted Class': [cnn_predicted_class, densenet_predicted_class, inceptionv3_predicted_class],
            'Probability': [cnn_probability, densenet_probability, inceptionv3_probability]
        }
        df = pd.DataFrame(data)

        st.image(image, caption='Uploaded Image', use_column_width=True)
        st.subheader('Predictions:')
        st.dataframe(df)

        cnn_accuracy = cnn_probability * 100
        densenet_accuracy = densenet_probability * 100
        inceptionv3_accuracy = inceptionv3_probability * 100

        accuracy_data = {
            'Model': ['CNN', 'densenet', 'inceptionv3'],
            'Accuracy': [cnn_accuracy, densenet_accuracy, inceptionv3_accuracy]
        }
        accuracy_df = pd.DataFrame(accuracy_data)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

        ax1.pie(accuracy_df['Accuracy'], labels=accuracy_df['Model'], autopct='%1.1f%%', startangle=90)
        ax1.axis('equal')
        ax1.set_title('Accuracy Comparison')

        remaining_percent = 100 - accuracy_df['Accuracy']
        ax2.bar(accuracy_df['Model'], remaining_percent)
        ax2.set_xlabel('Model')
        ax2.set_ylabel('Remaining Percentage')
        ax2.set_title('Remaining Percentage Comparison')

        st.subheader('Accuracy and Remaining Percentage Comparison')
        st.pyplot(fig)

        # Plot individual model predictions
        fig_individual, (ax3, ax4, ax5) = plt.subplots(1, 3, figsize=(15, 5))

        ax3.bar(class_names, cnn_prediction.squeeze())
        ax3.set_xlabel('Class')
        ax3.set_ylabel('Probability')
        ax3.set_title('CNN Model Predictions')

        ax4.bar(class_names, densenet_prediction.squeeze())
        ax4.set_xlabel('Class')
        ax4.set_ylabel('Probability')
        ax4.set_title('Densenet Model Predictions')

        ax5.bar(class_names, inceptionv3_prediction.squeeze())
        ax5.set_xlabel('Class')
        ax5.set_ylabel('Probability')
        ax5.set_title('InceptionV3 Model Predictions')

        st.subheader('Individual Model Predictions')
        st.pyplot(fig_individual)

        
# Run the Streamlit app
if __name__ == '__main__':
    app()

