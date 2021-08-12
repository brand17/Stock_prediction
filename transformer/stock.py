import tensorflow as tf
import numpy as np
import os
import random
import keras

print(tf.__version__)

d_model = 32
nsteps = 10
dummy_input = tf.random.uniform([8, 8], 0, 100, dtype=tf.int32)
from stock_data import stock_data

class stock():
  def __init__(self, cloud='', device='CPU'):
    if cloud  == 'kaggle':
      self.path_to_dataset = '/kaggle/input/accounting-dataset/'
      self.path_to_saved_model = '/kaggle/input/accounting-saved-model/'
      self.output_path = '/kaggle/working/'
      #path_to_embedding_model = '/kaggle/input/poetry-supervised-embedding-model/'
    else:
      path = ''
      if cloud == 'colab':
        path = '/content/drive/My_Drive/Colab_Notebooks/Accounting/'
        # if device == 'TPU':
        #   assert tf.__version__[:3] == '2.2'
      config = str(d_model) + '/' #'128-512/'
      self.path_to_dataset = path + 'data/'
      self.path_to_saved_model = path + 'saved/' + config
      self.output_path = path + 'saved/' + config
      #path_to_embedding_model = path + '../../embeddings/saved/' + config
    self.dataset = stock_data(self.path_to_dataset, nsteps, device, enc='cp1251', min_frequency=43)

  def create_model(self):
    dff = d_model * 4
    #dropout_rate = 0.1
    num_layers = 6
    num_heads = 8
    from transformers import TFRobertaForSequenceClassification, RobertaConfig
    vocab_size = len(self.dataset.index2word)#tokenizer_pt.vocab_size + 2
    config = RobertaConfig(
        vocab_size=vocab_size,
        hidden_size=d_model,
        intermediate_size=dff, 
        max_position_embeddings=514,
        num_attention_heads=num_heads,
        num_hidden_layers=num_layers,
        type_vocab_size=1,
        num_labels=len(self.dataset.index2word_tar)
    )  
    return TFRobertaForSequenceClassification(config)

  def train(self):
    import time
    from tf_utils import get_strategy, CustomSchedule, beam_search, create_logger
    strategy, dev = get_strategy()

    logger = create_logger(self.output_path)

    self.dataset.load_dataset('stock.pkl')
    learning_rate = CustomSchedule(d_model//6, 17000)
    #lr = learning_rate(259.)
    trainParameters={}
    import json
    #dummy_input = (dummy_input, dummy_input)

    @tf.function()
    def step_fn(input, labels):
      #print('retracing')
      with tf.GradientTape() as tape:
        logits = self.model(input, training=True)
        loss = tf.keras.losses.sparse_categorical_crossentropy(
            labels, logits, from_logits=True)
        loss = tf.reduce_mean(loss, -1)
        loss = tf.nn.compute_average_loss(loss, global_batch_size=batch_size)
      grads = tape.gradient(loss, self.model.trainable_variables)
      optimizer.apply_gradients(list(zip(grads, self.model.trainable_variables)))
      training_loss.update_state(loss * strategy.num_replicas_in_sync)

    @tf.function()
    def eval_step_fn(input, labels):
      logits = self.model(input)
      loss = tf.keras.losses.sparse_categorical_crossentropy(
          labels, logits, from_logits=True)
      loss = tf.reduce_mean(loss, -1)
      loss = tf.nn.compute_average_loss(loss, global_batch_size=batch_size)
      eval_loss.update_state(loss * strategy.num_replicas_in_sync)
      eval_accuracy.update_state(labels, logits)

    with strategy.scope():
      from transformers.optimization_tf import AdamWeightDecay
      optimizer = AdamWeightDecay(learning_rate,epsilon=1e-6,weight_decay_rate=0.01)
      optimizer._HAS_AGGREGATE_GRAD = False
      # from supervised_mmd_model import supervised_mmd
      self.model = self.create_model()
      # input = (
      #   tf.keras.layers.Input(shape=(None,), dtype=tf.int32), 
      #   tf.keras.layers.Input(shape=(None,), dtype=tf.int32)
      #   )
      # x = supervised_mmd(num_layers, d_model, num_heads, dff,
      #                           vocab_size, 
      #                           pe_input=514,#MAX_LENGTH, 
      #                           rate=dropout_rate, 
      #                           mmd_weight=10000.)(input)
      # model = tf.keras.Model(inputs=input, outputs=x)

      self.model.compile(
          loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
          optimizer=optimizer,
      )
      training_loss = tf.keras.metrics.Mean('training_loss', dtype=tf.float32)
      eval_loss = tf.keras.metrics.Mean('eval_loss', dtype=tf.float32)
      eval_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
        'eval_accuracy', dtype=tf.float32)
      # model.layers[-1].encoder.trainable = False
      import pickle
      if not os.path.isfile(self.path_to_saved_model+'trainParameters.txt'):
        print('creating model...')
        trainParameters={'bestLoss':float('inf'), 'iter':0}
        # from embedding_model import roberta_mmd
        # embModel = roberta_mmd(num_layers, d_model, num_heads, dff,
        #                         vocab_size, 
        #                         pe_input=514,#MAX_LENGTH, 
        #                         rate=dropout_rate, 
        #                         mmd_weight=10000.)
        # embModel(dummy_input)
        # embModel.load_weights(path_to_embedding_model + 'saved.h5')
        # model.layers[-1].encoder.set_weights(embModel.layers[0].get_weights())

        # batch_size = 8
        # strategy.run(step_fn, args=((dummy_input, dummy_input), dummy_input[:,0]))
        # model.summary()
      else:
        print('loading model...')
        trainParameters = json.load(open(self.path_to_saved_model+"trainParameters.txt"))
        # hist = model.fit(dummy_input, dummy_input, verbose=0)
        batch_size = 8
        strategy.run(step_fn, args=((dummy_input, dummy_input), dummy_input[:,0]))
        self.model.load_weights(self.path_to_saved_model+'saved.h5')
        with open(self.path_to_saved_model+'optimizer.pkl', 'rb') as f:
            weight_values = pickle.load(f)
        optimizer.set_weights(weight_values)

    #a = model(dummy_input, True)
    firstIter = trainParameters['iter']
    start = time.time()
    print('training started...')
    for i in range(firstIter, 1000000, nsteps):
      data = self.dataset.next()
      # print(tf.shape(input)[1])
      # print(i)
      # x = tf.random.uniform((8, 24), 1, 100, tf.int32)
      # sentences = ((x, x), x)
      # data = tf.data.Dataset.from_tensor_slices(sentences).batch(8)
      # for a in data:
      #   a = model(a)
      # batch_size = tf.cast(data._batch_size, tf.float32)
      # data = strategy.experimental_distribute_dataset(data)
      # hist = model.fit(data, verbose=0)
      
      training_loss.reset_states()
      # training_mmd_loss.reset_states()
      for input, labels in data:
        batch_size = tf.cast(tf.shape(labels)[0], tf.float32)
        # step_fn(input, labels)
        #print(tf.shape(input)[1])
        strategy.run(step_fn, args=(input, labels))

      if (i + nsteps) % 1000 == 0:
        # hist_eval = model.evaluate(dataset.validation_data, verbose=0)
        # evaluation_loss = evaluate()
        eval_loss.reset_states()
        eval_accuracy.reset_states()
        for input, labels in self.dataset.validation_data:
          batch_size = tf.cast(tf.shape(labels)[0], tf.float32)
          #eval_step_fn(input, labels)
          strategy.run(eval_step_fn, args=(input, labels))
        evaluation_loss = float(eval_loss.result())

        inaccuracy = 1. - float(eval_accuracy.result()) 
        r = optimizer._decayed_lr(tf.float32)
        out = 'Epoch {} Loss {:.6f} Inaccuracy {:.6f} Val loss {:.8f} time: {:.0f} lr: {:.6f}'.format(
            i + nsteps,
            round(float(training_loss.result()), 6),
            round(inaccuracy, 6),
            round(evaluation_loss, 6),
            time.time() - start,
            r,
            )
        print(out)
        logger.warning(out)

        if inaccuracy <= trainParameters['bestLoss']:
            print('saving...')
            self.model.save_weights(self.output_path+'saved.h5')
            weight_values = optimizer.get_weights()
            with open(self.output_path+'optimizer.pkl', 'wb') as f:
                pickle.dump(weight_values, f)
            trainParameters['bestLoss'] = inaccuracy
            trainParameters['iter'] = i + nsteps
            json.dump(trainParameters, open(self.output_path+"trainParameters.txt",'w'))

  def generate(self, sentence, max_length=80):
    input_tensor = self.convert_text_to_indices(sentence)
    enc_output, tar, dec_padding_mask = self.model.layers[-1].get_encoder_output(input_tensor)
    decoded_words = []
    for _ in range(max_length):
        #decoder_output = transf(input_tensor)
        decoder_output = self.model.layers[-1].get_decoder_output(enc_output, tar, dec_padding_mask) #+ self.bias
        decoder_output = decoder_output[:,-1]
        m = tf.math.top_k(decoder_output[0], 3)
        for i in range(3):
          predicted_id = m[1][i]
          if predicted_id != self.dataset.UNK_ACC_token and predicted_id != self.dataset.UNK_token:
            break
        if predicted_id == self.dataset.PAD_token:
            decoded_words.append('<EOS>')
            break
        else:
            decoded_words.append(self.dataset.index2word[predicted_id.numpy()])
        tar = tf.concat((tar, tf.reshape(predicted_id,(1,-1))), 1)
    return ' '.join(list(decoded_words))

  def convert_text_to_indices(self, sentence):
    sentence = self.dataset.bpe.process_line(sentence)
    enc_input = [self.dataset.get_word_index(word) for word in sentence.split()] #+ [self.dataset.PAD_token] * 4
    # dec_input = [self.dataset.SOS_token] #+ enc_input
    enc_input = tf.convert_to_tensor(enc_input)
    input_tensor = tf.reshape(enc_input,(1,-1))
    return input_tensor

  def excel_it(self, ws):
    cols = ws['B:E']
    for sent, dr_cl, cr_cl in zip(cols[0], cols[2], cols[3]):
      if sent.value != None:
        client = self.dataset.get_client_from_cell(dr_cl)
        if client == None:
          continue
        if client == 'Филиал "Корпоративный" ПАО "Совкомбанк" (Расчетный)':
          client = self.dataset.get_client_from_cell(cr_cl)
        yield sent.value + " <SEP> " + client

  def predict_to_excel(self):
    self.load_model()
    import openpyxl
    name = 'Карточка счета 51 за 1 quarter of 2021 Общество с ограниченной ответственностью  Трейд Хаус Компани.xlsx'
    wb = openpyxl.load_workbook(name)
    ws = wb.active
    for ind, inp in enumerate(self.excel_it(ws)):
      if ind < 1:
        continue
      inp = self.convert_text_to_indices(inp)
      account = self.predict(inp)
      cell = 'P' + str(ind + 8)
      ws[cell] = account
    wb.save(name)

  def predict(self, inp):
    output = self.model(inp)
    cat = tf.argmax(output, -1).numpy().squeeze((-1))[0]
    account = self.dataset.index2word_tar[cat]
    return account

  def load_model(self):
    self.model = self.create_model()
    self.model((dummy_input, dummy_input), dummy_input[:,0])
    self.model.load_weights(self.path_to_saved_model+'saved.h5')

  def save_evaluation_results(self):
    self.load_model()
    self.dataset.load_dataset('accounting.pkl')
    freq = [0] * len(self.dataset.index2word_tar)
    for v in self.dataset.buckets.values():
      if v.size != 0:
        for l in v[:, -1].tolist():
          freq[l] += 1
    import openpyxl
    name = 'analysis_validation.xlsx' #'Карточка счета 51 за January 2021 Общество с ограниченной ответственностью  Трейд Хаус Компани .xlsx'
    wb = openpyxl.load_workbook(name)
    ws = wb['Sheet2']
    output = self.model.predict(self.dataset.validation_data)[0]
    cat = tf.argmax(output, -1).numpy()
    probs = tf.nn.softmax(output)
    for ind, (inp, labels) in enumerate(self.dataset.validation_data.unbatch().batch(1)):
      best = cat[ind]
      account = self.dataset.index2word_tar[best]
      sent = []
      for i in inp['input_ids'][0].numpy().tolist():
        if i == self.dataset.PAD_token:
          break
        s = self.dataset.index2word[i]
        sent += [s]
      ws['A' + str(ind + 2)] = ' '.join(sent)
      ws['B' + str(ind + 2)] = account
      ws['C' + str(ind + 2)] = self.dataset.index2word_tar[labels[0].numpy()]
      ws['D' + str(ind + 2)] = probs[ind][best].numpy()
    # for ind, f in enumerate(freq):
    #   ws['G' + str(ind + 2)] = self.dataset.index2word_tar[ind]
    #   ws['H' + str(ind + 2)] = f
      
    wb.save(name)
    
# if __name__ == '__main__':
if 'kaggle' in globals():
  s = stock('kaggle')
elif 'colab' in globals():
  s = stock('colab')
else:
  s = stock()
s.train()