from keras.preprocessing import image
from keras.models import load_model
from keras.applications.vgg16 import preprocess_input
import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt


# Loading the saved model
model = load_model('our_model.h5')

# Below image is used to test the model
img_path = 'image.jpeg'

# Image loading and preprocessing
img = image.load_img(img_path, target_size=(224, 224))
imagee = image.img_to_array(img)
imagee = np.expand_dims(imagee, axis=0)
img_data = preprocess_input(imagee)

# Predicting if person is affected or not
prediction = model.predict(img_data)
if prediction[0][0] > prediction[0][1]:
    print('Person is safe.')
else:
    print('Person is affected with Pneumonia.')
print(f'Predictions: {prediction}')

# Grad-CAM Implementation
def generate_gradcam_heatmap(model, image_array, last_conv_layer_name="block5_conv3", pred_index=None):
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(image_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.reduce_max(heatmap)
    return heatmap.numpy()

# Heatmap generation
heatmap = generate_gradcam_heatmap(model, img_data, last_conv_layer_name="block5_conv3")

# Displaying the Grad-CAM heatmap on the original image
plt.figure(figsize=(10, 10))

# Original image
plt.subplot(1, 2, 1)
plt.imshow(img)
plt.title("Original Image")

# Applying heatmap to the original image
heatmap_resized = cv2.resize(heatmap, (img_data.shape[2], img_data.shape[1]))
heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
superimposed_img = cv2.addWeighted(np.uint8(imagee[0] * 255), 0.6, heatmap_colored, 0.4, 0)

# Plotting Grad-CAM image
plt.subplot(1, 2, 2)
plt.imshow(superimposed_img)
plt.title("Grad-CAM")

plt.show()
