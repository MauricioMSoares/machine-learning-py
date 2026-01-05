## Exporting the model
```python
import tensorflow as tf
import keras


# Train a model
model = keras.Sequential()
model.compile([...])kv
model.fit(x_train, y_train, epochs=10)

# Export the model
keras.saving.save_model(model, "model/1/")
```

## Run TensorFlow Serving
```bash
tensorflow_model_server --port=8500 --rest_api_port=8501 --model_name=example --model_base_path=/path/to/model/
```

## Making inference requests
```python
import requests
import json


# Define the input data
data = {"instances": [[...], [...], ...]}

# Send the inference request
response = requests.post("http://localhost:8501/v1/models/example:predict", json=data)

# Parse the response
output = json.loads(response.text)["predictions"]
```

## Converting a TensorFlow model
```bash
tensorflowjs_converter --input_format=tf_saved_model --output_format=tfjs_graph_model /path/to/model/ /path/to/tfjs_model/
```

## Loading a Tensorflow.js model
```javascript
const model = await tf.loadGraphModel("/path/to/tfjs_model/model.json")
```

## Making Inference Request
```javascript
const input = tf.tensor2d([[...], [...], ...])
const output = model.predict(input)
```