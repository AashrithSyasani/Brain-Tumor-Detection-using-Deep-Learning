
# import tensorflow as tf
# import numpy as np
# from tensorflow.keras.preprocessing import image
# from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# IMG_SIZE = 224

# model = tf.keras.models.load_model("best_brain_tumor_model.keras")

# class_labels = ["notumor", "tumor"]

# def predict_image(img_path):

#     img = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
#     img_array = image.img_to_array(img)

#     img_array = preprocess_input(img_array)
#     img_array = np.expand_dims(img_array, axis=0)

#     prediction = model.predict(img_array, verbose=0)[0]

#     predicted_index = np.argmax(prediction)
#     predicted_class = class_labels[predicted_index]

#     confidence = prediction[predicted_index] * 100

#     if predicted_class == "tumor":
#         result = "Tumor Detected"
#     else:
#         result = "No Tumor Detected"

#     print("\nPrediction Result")
#     print("---------------------")
#     print("Result      :", result)
#     # print("Confidence  : {:.2f}%".format(confidence))


# if __name__ == "__main__":

#     # predict_image("dataset_final2/Testing/notumor/Tr-no_324.jpg")
#     # predict_image("dataset_final2/Testing/notumor/Tr-no_346.jpg")
#     # predict_image("dataset_final2/Testing/notumor/Tr-no_5.jpg")
#     # predict_image("dataset_final2/Testing/notumor/Tr-no_17.jpg")
#     # predict_image("dataset_final2/Testing/tumor/Tr-aug-me_45.jpg")
#     # predict_image("dataset_final2/Testing/tumor/Tr-aug-me_10.jpg")
#     # predict_image("dataset_final2/Testing/tumor/Tr-aug-me_60.jpg")
#     predict_image("dataset/test/yes/Y77.jpg")


