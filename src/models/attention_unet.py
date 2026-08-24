import tensorflow as tf
from tensorflow.keras import layers, models, backend as K

def dice_coef(y_true, y_pred, smooth=1.0):
    """
    Dice Similarity Coefficient calculation.
    """
    y_true_f = K.flatten(K.cast(y_true, 'float32'))
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)

def focal_tversky_loss(y_true, y_pred, alpha=0.7, beta=0.3, gamma=0.75, smooth=1.0):
    """
    Focal Tversky Loss to heavily penalize false negatives for small lung nodules.
    alpha: weight for false negatives (recall emphasis)
    beta: weight for false positives
    gamma: focal parameter focusing on hard examples
    """
    y_true_f = K.flatten(K.cast(y_true, 'float32'))
    y_pred_f = K.flatten(y_pred)
    
    true_pos = K.sum(y_true_f * y_pred_f)
    false_neg = K.sum(y_true_f * (1.0 - y_pred_f))
    false_pos = K.sum((1.0 - y_true_f) * y_pred_f)
    
    tversky = (true_pos + smooth) / (true_pos + alpha * false_neg + beta * false_pos + smooth)
    return K.pow((1.0 - tversky), gamma)

def combo_loss(y_true, y_pred):
    """
    Hybrid loss: Binary Cross-Entropy + Focal Tversky Loss.
    """
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    ftl = focal_tversky_loss(y_true, y_pred)
    return bce + ftl

def attention_gate(g, s, num_filters):
    """
    Attention Gate Block.
    g: Gating signal from deeper layer
    s: Skip connection feature map from encoder
    num_filters: number of output channels
    """
    # Gating signal conv
    Wg = layers.Conv2D(num_filters, (1, 1), padding='same')(g)
    Wg = layers.BatchNormalization()(Wg)
    
    # Skip connection conv
    Ws = layers.Conv2D(num_filters, (1, 1), padding='same')(s)
    Ws = layers.BatchNormalization()(Ws)
    
    # Add gating and skip features
    out = layers.Activation('relu')(layers.add([Wg, Ws]))
    out = layers.Conv2D(1, (1, 1), padding='same')(out)
    out = layers.BatchNormalization()(out)
    psi = layers.Activation('sigmoid')(out)
    
    # Weight skip connection by attention coefficients
    return layers.multiply([s, psi])

def conv_block(input_tensor, num_filters):
    """
    Double convolution block with BatchNormalization and ReLU.
    """
    x = layers.Conv2D(num_filters, (3, 3), padding='same')(input_tensor)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    x = layers.Conv2D(num_filters, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    return x

def build_attention_unet(input_shape=(512, 512, 1), initial_filters=32):
    """
    Builds a 2D Attention U-Net architecture for CT lung nodule segmentation.
    """
    inputs = layers.Input(shape=input_shape, name="ct_input")
    
    # --- Encoder ---
    c1 = conv_block(inputs, initial_filters)
    p1 = layers.MaxPooling2D((2, 2))(c1)
    
    c2 = conv_block(p1, initial_filters * 2)
    p2 = layers.MaxPooling2D((2, 2))(c2)
    
    c3 = conv_block(p2, initial_filters * 4)
    p3 = layers.MaxPooling2D((2, 2))(c3)
    
    c4 = conv_block(p3, initial_filters * 8)
    p4 = layers.MaxPooling2D((2, 2))(c4)
    
    # --- Bottleneck ---
    bridge = conv_block(p4, initial_filters * 16)
    
    # --- Decoder with Attention Gates ---
    d4 = layers.Conv2DTranspose(initial_filters * 8, (2, 2), strides=(2, 2), padding='same')(bridge)
    a4 = attention_gate(g=d4, s=c4, num_filters=initial_filters * 8)
    u4 = layers.concatenate([d4, a4])
    c5 = conv_block(u4, initial_filters * 8)
    
    d3 = layers.Conv2DTranspose(initial_filters * 4, (2, 2), strides=(2, 2), padding='same')(c5)
    a3 = attention_gate(g=d3, s=c3, num_filters=initial_filters * 4)
    u3 = layers.concatenate([d3, a3])
    c6 = conv_block(u3, initial_filters * 4)
    
    d2 = layers.Conv2DTranspose(initial_filters * 2, (2, 2), strides=(2, 2), padding='same')(c6)
    a2 = attention_gate(g=d2, s=c2, num_filters=initial_filters * 2)
    u2 = layers.concatenate([d2, a2])
    c7 = conv_block(u2, initial_filters * 2)
    
    d1 = layers.Conv2DTranspose(initial_filters, (2, 2), strides=(2, 2), padding='same')(c7)
    a1 = attention_gate(g=d1, s=c1, num_filters=initial_filters)
    u1 = layers.concatenate([d1, a1])
    c8 = conv_block(u1, initial_filters)
    
    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid', name="nodule_mask")(c8)
    
    model = models.Model(inputs=[inputs], outputs=[outputs], name="Attention_UNet_Lung_Nodule")
    return model

if __name__ == "__main__":
    model = build_attention_unet(input_shape=(512, 512, 1))
    model.summary()
    print("Attention U-Net successfully constructed!")
