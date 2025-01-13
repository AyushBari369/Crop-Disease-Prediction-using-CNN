from flask import Flask, request, render_template
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
from io import BytesIO

app = Flask(__name__)

# Load the trained model
model = load_model('potatoesh6.keras')  # Make sure this path is correct

# Define the class names corresponding to the model's output
class_names = ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']

# Prepare the image for prediction
def prepare_image(image, target_size=(256, 256)):
    image = image.resize(target_size)
    image = img_to_array(image) / 255.0
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

@app.route("/", methods=["GET", "POST"])
def upload_predict():
    if request.method == "POST":
        image_file = request.files.get("image")
        if image_file:
            try:
                # Ensure the file is readable
                image = load_img(BytesIO(image_file.read()))  # Use BytesIO for in-memory file
                image = prepare_image(image)

                # Print shape of the prepared image
                print(f"Prepared image shape: {image.shape}")

                # Perform prediction
                prediction = model.predict(image)
                print(f"Model prediction: {prediction}")

                predicted_class = class_names[np.argmax(prediction[0])]
                print(f"Predicted class: {predicted_class}")

                return render_template("index.html", prediction=predicted_class)
            except Exception as e:
                return f"An error occurred: {str(e)}"

    return render_template("index.html", prediction=None)

@app.route("/predict", methods=["POST"])
def predict():
    image_file = request.files.get("image")
    if image_file:
        try:
            # Ensure the file is readable
            image = load_img(BytesIO(image_file.read()))  # Use BytesIO for in-memory file
            image = prepare_image(image)

            # Print shape of the prepared image
            print(f"Prepared image shape: {image.shape}")

            # Perform prediction
            prediction = model.predict(image)
            print(f"Model prediction: {prediction}")

            predicted_class = class_names[np.argmax(prediction[0])]
            print(f"Predicted class: {predicted_class}")

            return render_template("index.html", prediction=predicted_class)
        except Exception as e:
            return f"An error occurred: {str(e)}"

    return render_template("index1.html", prediction=None)

if __name__ == "__main__":
    app.run(debug=True)

