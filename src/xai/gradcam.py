import numpy as np
import tensorflow as tf
import cv2

class GradCAM:
    """
    Grad-CAM (Gradient-weighted Class Activation Mapping) for medical imaging explainability.
    Generates visual heatmaps to show which regions influenced the deep learning prediction.
    """
    def __init__(self, model, last_conv_layer_name=None):
        self.model = model
        self.last_conv_layer_name = last_conv_layer_name
        
        if self.last_conv_layer_name is None:
            self.last_conv_layer_name = self._find_last_conv_layer()

    def _find_last_conv_layer(self):
        """
        Automatically finds the deepest 4D convolutional layer.
        """
        for layer in reversed(self.model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D) or "conv" in layer.name.lower() or "top_conv" in layer.name.lower():
                return layer.name
        # Fallback to model layers
        for layer in reversed(self.model.layers):
            if hasattr(layer, 'output_shape') and len(layer.output_shape) == 4:
                return layer.name
        raise ValueError("Could not find a 4D convolutional layer in the model.")

    def compute_heatmap(self, img_array, eps=1e-8):
        """
        Computes Grad-CAM heatmap for a single image array (shape: (1, H, W, C)).
        """
        grad_model = tf.keras.models.Model(
            inputs=[self.model.inputs],
            outputs=[self.model.get_layer(self.last_conv_layer_name).output, self.model.output]
        )

        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            # Binary or multiclass target loss
            if predictions.shape[-1] == 1:
                loss = predictions[0]
            else:
                top_pred_index = tf.argmax(predictions[0])
                loss = predictions[:, top_pred_index]

        # Gradients of the predicted class with respect to the output feature map
        grads = tape.gradient(loss, conv_outputs)
        
        # Pooled gradients across spatial dimensions
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        # Weight the feature map by the pooled gradients
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)

        # Apply ReLU to retain only features that positively influence the target class
        heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + eps)
        return heatmap.numpy()

    def overlay_heatmap(self, heatmap, original_img, alpha=0.4, colormap=cv2.COLORMAP_JET):
        """
        Overlays the computed heatmap onto the original image.
        original_img: uint8 image (H, W, 3) or (H, W) in range [0, 255]
        """
        if len(original_img.shape) == 2:
            original_img = cv2.cvtColor(original_img, cv2.COLOR_GRAY2RGB)
        elif original_img.shape[-1] == 1:
            original_img = cv2.cvtColor(original_img[:, :, 0], cv2.COLOR_GRAY2RGB)

        # Resize heatmap to match image size
        heatmap_resized = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
        heatmap_colored = np.uint8(255 * heatmap_resized)
        heatmap_colored = cv2.applyColorMap(heatmap_colored, colormap)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

        # Superimpose
        superimposed_img = np.uint8(alpha * heatmap_colored + (1 - alpha) * original_img)
        return superimposed_img, heatmap_resized
