import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model


def generate_gradcam(
        model,
        image_path,
        output_path
):

    img = cv2.imread(image_path)

    img = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2RGB
    )

    original_img = cv2.resize(
        img,
        (224, 224)
    )

    img_array = np.expand_dims(
        original_img.astype(np.float32),
        axis=0
    )

    base_model = model.get_layer(
        "densenet201"
    )

    feature_model = Model(
        inputs=base_model.input,
        outputs=base_model.get_layer(
            "conv5_block32_concat"
        ).output
    )

    img_tensor = tf.convert_to_tensor(
        img_array,
        dtype=tf.float32
    )

    with tf.GradientTape() as tape:

        features = feature_model(
            img_tensor,
            training=False
        )

        tape.watch(features)

        x = model.layers[3](features)
        x = model.layers[4](x, training=False)
        x = model.layers[5](x, training=False)
        x = model.layers[6](x)
        x = model.layers[7](x, training=False)
        x = model.layers[8](x, training=False)
        predictions = model.layers[9](x)

        loss = predictions[:, 0]

    grads = tape.gradient(
        loss,
        features
    )

    weights = tf.reduce_mean(
        grads,
        axis=(1, 2)
    )

    cam = tf.reduce_sum(
        weights[:, None, None, :] * features,
        axis=-1
    )

    cam = tf.maximum(
        cam[0],
        0
    )

    heatmap = cam.numpy()

    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()

    heatmap = cv2.resize(
        heatmap,
        (224, 224)
    )

    heatmap = np.uint8(
        255 * heatmap
    )

    heatmap = cv2.applyColorMap(
        heatmap,
        cv2.COLORMAP_JET
    )

    overlay = cv2.addWeighted(
        cv2.cvtColor(
            original_img,
            cv2.COLOR_RGB2BGR
        ),
        0.6,
        heatmap,
        0.4,
        0
    )

    cv2.imwrite(
        output_path,
        overlay
    )

    return output_path