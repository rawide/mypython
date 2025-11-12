import tensorflow as tf
import numpy as np

def simple_quantization(input_model_path, output_model_path):
    """
    简单的PTQ量化函数
    """
    # 根据模型类型创建转换器
    if input_model_path.endswith('.h5'):
        # Keras .h5 模型
        model = tf.keras.models.load_model(input_model_path)
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
    elif input_model_path.endswith('.tflite'):
        # 已经是tflite模型
        converter = tf.lite.TFLiteConverter.from_saved_model(input_model_path)
    else:
        # SavedModel 格式
        converter = tf.lite.TFLiteConverter.from_saved_model(input_model_path)
    
    # 设置量化参数
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    # 转换并保存
    tflite_quant_model = converter.convert()
    
    with open(output_model_path, 'wb') as f:
        f.write(tflite_quant_model)
    
    print(f"量化完成！模型已保存至: {output_model_path}")

# 使用示例
if __name__ == "__main__":
    input_model = "vgg16_fp32.tflite"  # 您的FP32模型
    output_model = "vgg16_int8.tflite" # 输出INT8模型
    
    simple_quantization(input_model, output_model)
