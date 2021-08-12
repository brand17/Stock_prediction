import tensorflow as tf

from poetry_model import gelu, my_roberta, DecoderLayer
from transformers import RobertaConfig
from transformers import __version__ as transf_ver
assert transf_ver == '3.3.1'
def create_padding_mask(seq):
  seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
  
  # add extra dimensions to add the padding
  # to the attention logits.
  return seq  # (batch_size, 1, 1, seq_len)

def create_look_ahead_mask(size):
  mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
  return mask  # (seq_len, seq_len)

def create_masks(inp, tar):
  # Encoder padding mask
  enc_padding_mask = create_padding_mask(inp)
  
  # Used in the 2nd attention block in the decoder.
  # This padding mask is used to mask the encoder outputs.
  dec_padding_mask = create_padding_mask(inp)[:, tf.newaxis, tf.newaxis,:]

  #paddings = tf.constant([[0, 0,], [0, 0], [0, 0], [1, 0]])
  #dec_padding_mask = tf.pad(dec_padding_mask, paddings, "CONSTANT")
  
  # Used in the 1st attention block in the decoder.
  # It is used to pad and mask future tokens in the input received by 
  # the decoder.
  combined_mask = create_decoder_mask(tar)
  
  return enc_padding_mask, combined_mask, dec_padding_mask

def create_decoder_mask(tar):
  look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
  dec_target_padding_mask = create_padding_mask(tar)[:, tf.newaxis, tf.newaxis,:]
  combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)
  return combined_mask

class Decoder(tf.keras.layers.Layer):
  def __init__(self, num_layers, d_model, num_heads, dff, target_vocab_size,
               maximum_position_encoding, embeddings, rate=0.1):
    super(Decoder, self).__init__()

    self.d_model = d_model
    self.num_layers = num_layers
    self.embedding = embeddings

    self.dec_layers = [DecoderLayer(d_model, num_heads, dff, rate) 
                       for _ in range(num_layers)]
    self.dropout = tf.keras.layers.Dropout(rate)
    
  def call(self, x, enc_output, training, 
           look_ahead_mask, padding_mask):

    #seq_len = tf.shape(x)[1]
    attention_weights = {}
    
    x = self.embedding(x)  # (batch_size, target_seq_len, d_model)
    #x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
    #x += self.pos_encoding[:, :seq_len, :]
    
    #x = self.dropout(x, training=training)

    for i in range(self.num_layers):
      x, block1, block2 = self.dec_layers[i](x, enc_output, training,
                                             look_ahead_mask, padding_mask)
      
      attention_weights['decoder_layer{}_block1'.format(i+1)] = block1
      attention_weights['decoder_layer{}_block2'.format(i+1)] = block2
    
    # x.shape == (batch_size, target_seq_len, d_model)
    return x, attention_weights

class Transformer(tf.keras.Model):
  def __init__(self, num_layers, d_model, num_heads, dff, vocab_size, 
               pe_input, rate=0.1):
    super(Transformer, self).__init__()

    config = RobertaConfig(
        vocab_size=vocab_size,
        hidden_size=d_model,
        intermediate_size=dff, 
        max_position_embeddings=pe_input,
        num_attention_heads=num_heads,
        num_hidden_layers=num_layers,
        type_vocab_size=1,
    )  
    roberta = my_roberta(config)
    self.decoder = Decoder(num_layers, d_model, num_heads, dff, 
                           vocab_size, pe_input, roberta.embeddings, rate)

    self.act = tf.keras.layers.Activation(gelu)
    self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6, name="layer_norm")
    self.embeddings = roberta.embeddings
    self.encoder = roberta
    #self.bias = self.add_weight(shape=(input_vocab_size,), initializer="zeros", trainable=True, name="bias")
    
  def call(self, inp, training=False):
    inp, tar = inp
    enc_padding_mask, look_ahead_mask, dec_padding_mask = create_masks(inp, tar)

    enc_output = self.encoder((inp, enc_padding_mask), training=training)[0]  # (batch_size, inp_seq_len, d_model)
    
    dec_output = self.decoder(
        tar, enc_output, training, look_ahead_mask, dec_padding_mask)[0]
    
    x = self.act(dec_output)
    x = self.layer_norm(x)
    x = self.embeddings(x, mode='linear') #+ self.bias
    
    return x


