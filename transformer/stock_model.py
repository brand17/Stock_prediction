import tensorflow as tf

from transformers import RobertaConfig, TFRobertaModel

class Transformer(tf.keras.Model):
  def __init__(self):
    super(Transformer, self).__init__()

    self.roberta = TFRobertaModel.from_pretrained('roberta-base')
    self.roberta.trainable = False
    self.regression_head = tf.keras.layers.Dense(1)
    
  def call(self, inp, training=False):
    x = self.roberta(inp).last_hidden_state[:, 0]
    x = self.regression_head(x)
    
    return x


