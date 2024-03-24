import streamlit as st

from PIL import Image
import numpy as np
from io import BytesIO
#from sklearn.neighbors import KNeighborsClassifier
import joblib

#knn.fit(X_train_scaled, y_train)
model = joblib.load('SVM_test_trained') 
nav = st.sidebar.selectbox("Navigation Menu",["Prediction", "Chatting"])
 
if nav == "Chatting":

    with st.sidebar:

        messages = st.container(height=300)

        if prompt := st.chat_input("Say something"):

            messages.chat_message("user").write(prompt)

            messages.chat_message("assistant").write(f"Help: {prompt}")

if nav == "Prediction":

    def predict_number(image, model):

    # Convert image to grayscale
        image = image.convert('L')

    # Apply a threshold to the image to binarize
    # we can adjust 'threshold' depending on the image's brightness/contrast
        threshold = 200
        image = image.point(lambda p: p > threshold and 255)

    # Invert colors of the image
        image = Image.eval(image, lambda x: 255 - x)

    # Resize image to 28x28
        image = image.resize((28, 28), Image.Resampling.LANCZOS)

    # Convert image to numpy array and ensure it's uint8
        img_array = np.array(image, dtype=np.uint8)

    # Flatten the image array
        img_flat = img_array.flatten()

    # Normalize pixel values
        img_scaled = img_flat / 255.0

    # Reshape the array to match the input shape of the model
        img_reshaped = img_scaled.reshape(1, -1)

    # Make prediction
        prediction = model.predict(img_reshaped)

        return prediction[0]

#Streamlit app

    def main():

        st.title('Digit Recognition App')

        st.write('Upload an image of a digit (0-9) or capture using webcam for prediction.')

        # Option to upload image or capture using webcam

        option = st.selectbox("Choose input option:", ("Upload Image", "Use Webcam"))

        if option == "Upload Image":

        # Upload image

            uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

            if uploaded_file is not None:

                # Convert the uploaded file to bytes

                file_bytes = uploaded_file.read()

            # Convert the bytes data to a PIL Image object

                image = Image.open(BytesIO(file_bytes))

                st.image(image, caption='Uploaded Image.', use_column_width=True)

            # Make prediction

                prediction = predict_number(image, model)

                st.write(f"Prediction: {prediction}")

        elif option == "Use Webcam":

            st.title("Webcam Input Example")

            frame =  st.camera_input("Take a picture")

            if frame is not None:

            # Convert frame to bytes

                frame_bytes = frame.read()

            # Convert frame bytes to PIL image

                pil_image = Image.open(BytesIO(frame_bytes))

            # Make prediction

                prediction = predict_number(pil_image, model)

                st.write(f"Prediction: {prediction}")

    if __name__ == '__main__':

        main()