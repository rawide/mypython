import tensorflow as tf, numpy as np

model = tf.keras.applications.VGG16(weights="imagenet", include_top=True)

def representative_dataset():
    for _ in range(64):  # 数量不必多，32~100 都可；你不追精度
        x = np.random.randint(0, 256, size=(1,224,224,3), dtype=np.uint8).astype(np.float32)
        x = tf.keras.applications.vgg16.preprocess_input(x)  # RGB->BGR & 减均值
        yield [x]

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
# 要求算子使用 INT8 内核
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
# 不设置 I/O 类型 => I/O 维持 float32（最稳）
tflite_int8_fp_io = converter.convert()
open("vgg16_int8_fp_io.tflite","wb").write(tflite_int8_fp_io)

