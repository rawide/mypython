import tensorflow as tf
import numpy as np

model = tf.keras.applications.ResNet50(weights="imagenet", include_top=True)

def representative_dataset():
    for _ in range(100):
        # 这里换成你自己的校准图像读取与预处理（224x224, RGB, [0,255]或[0,1]）
        data = np.random.randint(0, 256, size=(1,224,224,3), dtype=np.uint8)
        yield [data.astype(np.float32)]  # 如果用 uint8 输入也可相应调整

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
# 若要整网 int8 推理（含输入/输出），需指定：
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8

tflite_int8 = converter.convert()
open("resnet50_int8.tflite", "wb").write(tflite_int8)

