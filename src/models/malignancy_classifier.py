import tensorflow as tf
from tensorflow.keras import layers, models

def build_malignancy_classifier(input_shape=(128, 128, 3), num_classes=2, backbone="EfficientNetV2B0"):
    """
    Builds a Stage-2 Nodule Malignancy Classifier using Transfer Learning.
    Classes: 
      0: Benign (Non-cancerous)
      1: Malignant (Cancerous / Adenocarcinoma / Squamous)
    """
    inputs = layers.Input(shape=input_shape, name="nodule_patch_input")
    
    # Preprocessing layer for transfer learning backbones
    if backbone == "EfficientNetV2B0":
        base_model = tf.keras.applications.EfficientNetV2B0(
            include_top=False,
            weights="imagenet",
            input_tensor=inputs
        )
    elif backbone == "ResNet50":
        base_model = tf.keras.applications.ResNet50(
            include_top=False,
            weights="imagenet",
            input_tensor=inputs
        )
    else:
        # Lightweight Custom CNN backbone
        x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        
        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        
        x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.GlobalAveragePooling2D()(x)
        base_model = models.Model(inputs=inputs, outputs=x)
    
    # Fine-tuning: Freeze top layers initially if using ImageNet
    if hasattr(base_model, 'layers'):
        for layer in base_model.layers[:-15]:
            layer.trainable = False
        for layer in base_model.layers[-15:]:
            layer.trainable = True

    x = base_model.output
    if len(x.shape) > 2:
        x = layers.GlobalAveragePooling2D(name="classifier_gap")(x)
        
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    
    if num_classes == 2:
        outputs = layers.Dense(1, activation='sigmoid', name="malignancy_prob")(x)
    else:
        outputs = layers.Dense(num_classes, activation='softmax', name="subtype_probs")(x)
        
    model = models.Model(inputs=inputs, outputs=outputs, name=f"Malignancy_Classifier_{backbone}")
    return model

if __name__ == "__main__":
    clf = build_malignancy_classifier(input_shape=(128, 128, 3))
    clf.summary()
    print("Malignancy Classifier successfully created!")
