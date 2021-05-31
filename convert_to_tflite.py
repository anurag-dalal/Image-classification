import tensorflow as tf

model = tf.keras.models.load_model('detectv1.h5', custom_objects=None, compile=True, options=None)
# Convert the model.
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the model.
with open('model.tflite', 'wb') as f:
  f.write(tflite_model)