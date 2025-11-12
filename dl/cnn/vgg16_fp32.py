import tensorflow as tf

model = tf.keras.applications.VGG16(weights="imagenet", include_top=True)
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_bytes = converter.convert()
open("vgg16_fp32.tflite","wb").write(tflite_bytes)

