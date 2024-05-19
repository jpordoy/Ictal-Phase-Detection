import numpy as np
from keras.layers import Layer
from keras import backend as K



# Define the TemporalAttention class
class TemporalAttention(Layer):
    def __init__(self, **kwargs):
        super(TemporalAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert isinstance(input_shape, list)
        self.W = self.add_weight(name='W_temporal', 
                                 shape=(input_shape[0][-1], 1),
                                 initializer='glorot_uniform',
                                 trainable=True)
        super(TemporalAttention, self).build(input_shape)

    def call(self, inputs):
        assert isinstance(inputs, list)
        encoder_output_t, encoder_output_tp1 = inputs
        
        # Compute attention scores for timestep t
        e_t = K.squeeze(K.tanh(K.dot(encoder_output_t, self.W)), axis=-1)
        alpha_t = K.expand_dims(K.softmax(e_t, axis=-1))
        context_vector_t = K.sum(encoder_output_t * alpha_t, axis=1)

        # Compute attention scores for timestep t+1
        e_tp1 = K.squeeze(K.tanh(K.dot(encoder_output_tp1, self.W)), axis=-1)
        alpha_tp1 = K.expand_dims(K.softmax(e_tp1, axis=-1))
        context_vector_tp1 = K.sum(encoder_output_tp1 * alpha_tp1, axis=1)

        # Combine context vectors from both timesteps
        combined_context_vector = K.concatenate([context_vector_t, context_vector_tp1], axis=-1)

        return combined_context_vector

    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list)
        return input_shape[0][0], input_shape[0][-1] * 2  # Concatenating context vectors

# Create dummy data
encoder_output_t = np.random.rand(2, 10, 64)  # (samples, timesteps, features)
encoder_output_tp1 = np.random.rand(2, 10, 64)  # (samples, timesteps, features)

# Instantiate TemporalAttention layer
temporal_attention_layer = TemporalAttention()

# Call the layer with the dummy data
combined_context_vector = temporal_attention_layer([encoder_output_t, encoder_output_tp1])

# Print input of timestep t
print("Input of timestep t:")
print(encoder_output_t)

# Print input of timestep t+1
print("\nInput of timestep t+1:")
print(encoder_output_tp1)

# Print output of timestep t
print("\nOutput of timestep t:")
print(combined_context_vector[:, :64])

# Print output of timestep t+1
print("\nOutput of timestep t+1:")
print(combined_context_vector[:, 64:])
