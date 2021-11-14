if 'kaggle' in globals():
  cloud = 'kaggle'
elif 'colab' in globals():
  cloud = 'colab'
else:
  cloud = ''
# from math import sqrt
import pickle
from dateutil.relativedelta import relativedelta
import tensorflow as tf
import os

from market_data import market_data

from tf_utils import get_strategy, CustomSchedule, create_logger

print(tf.__version__)
import transformers
print(transformers.__version__)
from transformers import TFRobertaForSequenceClassification, RobertaConfig
import pandas as pd

# from scipy.stats import spearmanr

d_model = 32
dummy_input = (
  tf.random.uniform([8, 8], 0, 100, dtype=tf.int32),
  tf.random.uniform([8], 0.5, 1.5, dtype=tf.float32)
  )
from stock_data import stock_data
from utils import time_range

# def to_matrix(l, n):
#   return [l[i:i+n] for i in range(0, len(l), n)]

class cloud_path_initialilzer():
  def __init__(self, path_to_model, cloud=''):
    if cloud  == 'kaggle':
      self.path_to_dataset = '/kaggle/input/securities-dataset-by-dates/'
      self.path_to_saved_model = '/kaggle/input/securities-transf-saved-model-new/' + path_to_model
      self.output_path = '/kaggle/working/' + path_to_model
      self.path_to_market_data = '/kaggle/input/market-data/'
      # self.path_to_validation = self.path_to_saved_model
      self.path_to_validation = '/kaggle/input/securities-transf-validation/' + path_to_model
    else:
      path = ''
      self.path_to_dataset = "../download_from_CC/result" #path + 'data/'
      self.path_to_saved_model = path + 'saved/' + path_to_model
      self.output_path = path + 'saved/' + path_to_model
      # self.output_path = 'test_outputs/gross/' + path_to_model
      self.path_to_market_data = path + '../data/'
      self.path_to_validation = path + 'saved/' + path_to_model
    from pathlib import Path
    Path(self.output_path).mkdir(parents=True, exist_ok=True)

class test_securities(cloud_path_initialilzer):
  pretrained_model_name = 'roberta-base'
  def __init__(self, test_range, train_range, cloud=''):
    super().__init__(str(train_range.start) + '/', cloud)
    self.train_range, self.test_range = train_range, test_range

  def load_data(self, path_test_with_ret):
    # @tf.function()
    # def get_y_pred(i):
    #   output = self.model(i).logits
    #   return output

    from utils import check_file
    if check_file(path_test_with_ret):
      strategy, dev = get_strategy()
      assert dev != 'TPU', "Predict method not working with TPU - https://github.com/huggingface/transformers/issues/12202"
      self.init_dataset(dev)
      if dev != 'CPU':
        self.dataset.nsteps = 64
        # self.dataset.set_batch_sizes_based_on(128)
      with strategy.scope():
        self.load_model()
        with open(path_test_with_ret, 'w') as f:
          from time import time
          start = time()
          it = self.dataset.sentences_it(self.test_range, verbose=1, roundup=False)
          for sentence in self.dataset.load_batch(it):
            data = self.dataset.dataset_from_target(sentence[:1])
            data = strategy.experimental_distribute_dataset(data)
            # y_pred = []
            # for i in data:
            #   # per_replica_result = get_y_pred(i)
            #   per_replica_result = strategy.run(get_y_pred, args=(i,))
            #   try:
            #     per_replica_result = per_replica_result.values
            #   except AttributeError:
            #     pass
            #   y_pred.append(per_replica_result)
            # prediction = tf.concat(y_pred, 0)
            # prediction = tf.reshape(y_pred, [-1, 1])
            prediction = self.model.predict(data, verbose=0).logits # not working with TPU
            for s in zip(*sentence, prediction):
              f.write(str(s[4]) + '\t' + s[3] + '\t' + str(s[2]) + '\t' + str(s[1]) + '\t' + str(s[5][0]) + '\n')

            # for inp in data:
            #   # prediction = step_fn(inp)
            #   prediction = strategy.run(step_fn, args=(inp,)).logits.numpy()
            #   for s in zip(*sentence, prediction):
            #     f.write(str(s[4]) + '\t' + s[3] + '\t' + str(s[2]) + '\t' + str(s[1]) + '\t' + str(s[5][0]) + '\n')
            print(time() - start)
    df = pd.read_csv(
      path_test_with_ret, '\t', 
      names=['articleID', 'comp_short', 'publ_time', 'score', 'prediction'],
      parse_dates=[2]
      )
    df['publ_date'] = df.publ_time.dt.floor('d')
    # df = df.set_index('publ_time').between_time('00:00', '12:00')
    df = df.groupby(['articleID', 'comp_short', 'publ_date']).mean()
    return df

  # def test_strategy(self, sample_size=1, sign=1):
  #   self.df.prediction *= sign
  #   inds = self.df.prediction.groupby('publ_date', group_keys=False).nlargest(sample_size).index
  #   rets_full = self.df.loc[inds]
  #   rets_full = [(pd.Timestamp(d), c, r.prediction) for (d, c), r in rets_full.iterrows()]
  #   res = []
  #   for i in range(sample_size):
  #     rets = rets_full[i::sample_size]
  #     cash = self.test_sample(rets, sign)
  #     res += [cash]
  #   return sum(res) / len(res)

  # def test_sample(self, rets, sign=1):
  # # !!! error here - saturday-monday averaged instead of friday-sunday
  #   rets += [(rets[-1][0] + relativedelta(days=1), 'A', 1.)]
  #   work_days = [i[0] in self.quotes.index for i in rets]
  #   best = None
  #   cash = 100000
  #   invested = {'comp': 'A', 'quantity': 0, 'quote': 1}
  #   for comp2buy, wd in zip(rets, work_days):
  #     if best is None or comp2buy[2] > best[2]:
  #       best = comp2buy
  #     if not wd:
  #       continue
  #     quote = self.quotes.loc[comp2buy[0], invested['comp']]
  #     cash += (quote - invested['quote']) * invested['quantity']
  #     with open(self.output_path + '../output.txt', 'a') as f:
  #       f.write(str(comp2buy[0]) + '\t' + str(cash) + '\t' + str(invested['comp']) + '\t' + str(invested['quantity']) + '\n')
  #     quote = self.quotes.loc[comp2buy[0], best[1]]
  #     quantity = sign * cash // quote
  #     invested = {'comp': best[1], 'quantity': quantity, 'quote': quote}
  #     best = None
  #   return cash

  def test(self):
    if True:#os.path.exists(self.path_to_saved_model + 'saved.h5'):
      path_test_with_ret = self.output_path + 'test_return_forecast.txt'
      from market_data import load_market_data, get_spreads
      self.quotes = load_market_data(
        self.path_to_market_data + 'market_data.parquet',
        self.path_to_market_data + 'market_data.xlsx',
      )
      self.quotes = self.quotes.stack().to_frame('quote')
      self.quotes['next_day_quote'] = self.quotes.groupby(level=1)['quote'].shift(-1)
      self.spreads = get_spreads(self.path_to_market_data)
      articles_avg = self.load_data(path_test_with_ret)
      comps_avg = articles_avg.groupby(['comp_short']).mean() 
      best = comps_avg.loc[comps_avg['prediction'].idxmax()].name
      print('the best company -', best)
      # comps_avg.to_excel(self.output_path + 'output.xlsx', sheet_name="Sheet1", engine='openpyxl')
      self.df = articles_avg.groupby(['publ_date', 'comp_short']).mean()
      # min_max = self.df.groupby('publ_date').agg({'score' : ['min', 'max'], 'prediction' : ['min', 'max']})
      # min_max.to_csv(self.output_path + '../minmax.csv', mode='a', header=False)

      a = self.test_strategy(40, 1)
      # a = self.test_strategy(40, -1)
    else:
      print(self.path_to_saved_model + 'saved.h5', 'not found')
            
  def test_strategy(self, sample_size=1, sign=1):
    self.df.prediction *= sign 
    # self.df.prediction *= -sign # !!! choosing the worst predictions
    self.df = self.df.drop('score', axis=1)
    score_inds = self.df.index.unique(level=0) 
    holidays = ~score_inds.isin(self.quotes.index.levels[0])
    if holidays.any():
      import numpy as np
      to_avg = np.logical_or(holidays[1:], holidays[:-1])
      to_avg = np.concatenate([to_avg, holidays[-1:]])
      shifted = np.concatenate([[False], to_avg[:-1]])
      for_grouping = np.logical_xor(to_avg, shifted)
      groups = np.cumsum(for_grouping)
      groups_df = pd.DataFrame({'groups': groups}, index=score_inds)
      groups_df = groups_df[to_avg]
      merged = pd.merge(self.df, groups_df, left_on='publ_date', right_index=True)
      holidays_scores = merged.reset_index().groupby(['groups', 'comp_short']).agg({'publ_date': 'min', 'prediction': 'mean'})
      holidays_scores = holidays_scores.reset_index().drop('groups', axis=1).set_index(['publ_date', 'comp_short'])
      workday_scores = self.df[~self.df.index.get_level_values(level=0).isin(holidays_scores.index.levels[0])]
      self.df = pd.concat([workday_scores, holidays_scores])

    inds = self.df.prediction.groupby('publ_date', group_keys=False).nlargest(sample_size).index
    rets_full = self.df.loc[inds]
    self.quotes.index.names = ['publ_date', 'comp_short']
    merged = pd.merge(rets_full, self.quotes, left_index=True, right_index=True)
    merged = pd.merge(merged, self.spreads, left_index=True, right_index=True)
    costs = merged.quote * (1 + merged.spreads * 0.5) + 0.005
    sell = merged.next_day_quote * (1 - merged.spreads * 0.5) - 0.005
    merged['yield'] = sell / costs
    merged['pos'] = merged.groupby(['publ_date']).cumcount()
    # merged.groupby(['pos', 'publ_date', 'comp_short']).mean().to_excel('output2.xlsx')
    merged = merged.groupby(['pos', 'publ_date', 'comp_short'])['yield'].mean()
    merged.to_csv(self.output_path + '../output.csv', mode='a', header=False)
    # merged = merged.reset_index().set_index(['pos', 'publ_date'])
    # merged.to_excel(self.output_path + '../output.xlsx')
    # merged = merged.reset_index()[['yield', 'pos']].groupby(['pos']).prod()
    pass

  def init_model(self):
    config = RobertaConfig.from_pretrained(self.pretrained_model_name)
    config.num_labels = 1
    config.vocab_size = len(self.dataset.tok.vocab)
    self.model = TFRobertaForSequenceClassification(config)

  def load_model(self):
    self.init_model()
    self.model(dummy_input[0])
    self.model.load_weights(self.path_to_saved_model + 'saved.h5')

  def init_dataset(self, dev='CPU', nsteps=10):
    self.dataset = stock_data(
      self, nsteps,
      device=dev, 
      #companies=set(['MS'])
      )

class stock(test_securities):
  def load_embeddings(self):
    pass

  def train(self, fine_tune=False):
    import time
    strategy, dev = get_strategy()
    assert dev != 'GPU', 'train on TPU or CPU !!!'
    nsteps = 1
    self.init_dataset(dev, nsteps)
    self.dataset.prepare_for_training()
    eval_batch_size = tf.cast(self.dataset.validation_data._batch_size, tf.float32)
    self.dataset.validation_data = strategy.experimental_distribute_dataset(self.dataset.validation_data)

    trainParameters={}
    import json

    @tf.function()
    def step_fn(input, labels):
      with tf.GradientTape() as tape:
        output = self.model(input, training=True)
        loss = tf.keras.losses.MSE(
            labels, output.logits)
        loss = tf.nn.compute_average_loss(loss, global_batch_size=batch_size)
      grads = tape.gradient(loss, self.model.trainable_variables)
      optimizer.apply_gradients(list(zip(grads, self.model.trainable_variables)))
      training_loss.update_state(loss * strategy.num_replicas_in_sync)

    # def eval_acc_step_fn():
    #   @tf.function()
    #   def get_y_pred(i, labels):
    #     output = self.model(i).logits
    #     loss = tf.keras.losses.MSE(
    #         labels, output)
    #     loss = tf.nn.compute_average_loss(loss, global_batch_size=eval_batch_size)
    #     eval_loss.update_state(loss * strategy.num_replicas_in_sync)
    #     return output

    #   y_pred = []
    #   labels = []
    #   for i, l in self.dataset.validation_data: 
    #     # per_replica_result = get_y_pred(i, l)
    #     per_replica_result = strategy.run(get_y_pred, args=(i, l))
    #     try:
    #       per_replica_result = strategy.gather(per_replica_result, axis=0)
    #       l = l.values
    #     except AttributeError:
    #       pass
    #     y_pred.append(per_replica_result)
    #     labels += [l]
    #   y_pred = tf.concat(y_pred, 0)
    #   y_pred = tf.reshape(y_pred, [-1])
    #   labels = tf.concat(labels, 0)
    #   labels = tf.reshape(labels, [-1])

    #   comps = self.dataset.validation_data_comps[:tf.shape(y_pred)[0]]

    #   def groupby(y):
    #     d = {'col1': comps, 'col2': y}
    #     df = pd.DataFrame(data=d)
    #     df = df.groupby('col1').mean()
    #     return df

    #   df_pred = groupby(y_pred)
    #   df_fact = groupby(labels)
    #   spearman = spearmanr(df_pred.values, df_fact.values)

    #   return spearman.correlation

    @tf.function()
    def eval_step_fn(input, labels):
      output = self.model(input)
      loss = tf.keras.losses.MSE(
          labels, output.logits)
      loss = tf.nn.compute_average_loss(loss, global_batch_size=eval_batch_size)
      eval_loss.update_state(loss * strategy.num_replicas_in_sync)
      # eval_accuracy.update_state(labels, logits)

    with strategy.scope():
      self.init_model()

      class RsqrtSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
        def __call__(self, step):
          init_rate = 0.0003
          rate = tf.math.rsqrt(step + 1) * init_rate
          # flat_steps = 1
          # rate = tf.cond(step < flat_steps, lambda: init_rate, lambda: tf.math.rsqrt(step - flat_steps + 1) * init_rate)
          # mult = 4 / 1666 * step + 1
          # rate /= mult
          return rate

      # learning_rate = CustomSchedule(768, 17000)
      learning_rate = RsqrtSchedule()
      # learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(
      #   3e-4,
      #   decay_steps=1,
      #   decay_rate=0.997,
      #   )
      #lr = learning_rate(259.)
      from transformers.optimization_tf import AdamWeightDecay
      optimizer = AdamWeightDecay(learning_rate, epsilon=1e-6, weight_decay_rate=0.01)
      optimizer._HAS_AGGREGATE_GRAD = False
      self.model.compile(
          loss=tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE),
          optimizer=optimizer,
      )
      training_loss = tf.keras.metrics.Mean('training_loss', dtype=tf.float32)
      eval_loss = tf.keras.metrics.Mean('eval_loss', dtype=tf.float32)
      # model.layers[-1].encoder.trainable = False
      import pickle
      if not os.path.isfile(self.path_to_saved_model + 'trainParameters.txt'):
        print('creating model...')
        trainParameters={'bestLoss':float('inf'), 'iter':0}
        self.model(dummy_input[0])
        self.load_embeddings()
        # self.model.summary()
        # c = self.model.roberta.embeddings.count_params()
        # c = self.model.roberta.encoder.count_params()
      else:
        print('loading model...')
        trainParameters = json.load(open(self.path_to_saved_model + "trainParameters.txt"))
        # hist = model.fit(dummy_input, dummy_input, verbose=0)
        batch_size = 8
        if fine_tune:
          # step_fn(*dummy_input)
          self.model.roberta.trainable = True
          strategy.run(step_fn, args=(dummy_input[0], dummy_input[1]))
          self.model.load_weights(self.path_to_saved_model + 'saved.h5')
          optimizer.iterations.assign(trainParameters['iter'])
          # self.model.summary()
        else:
          strategy.run(step_fn, args=(dummy_input[0], dummy_input[1]))
          self.model.load_weights(self.path_to_saved_model + 'saved.h5')
          optimizer.iterations.assign(trainParameters['iter'])
          with open(self.path_to_saved_model + 'optimizer.pkl', 'rb') as f:
              weight_values = pickle.load(f)
          optimizer.set_weights(weight_values)

    firstIter = trainParameters['iter']
    start = time.time()
    print('training started...')
    min_tr_loss = 0.000565
    min_eval_loss = 0.00091
    for _ in range(1):
      for i, data in zip(
        range(firstIter, 4000, nsteps), 
        self.dataset.next(verbose=1, roundup=True, max_batches_per_file=35)
        # self.dataset.next_randomly(verbose=1, roundup=True)
        ):
        batch_size = tf.cast(data._batch_size, tf.float32)
        data = strategy.experimental_distribute_dataset(data)
        training_loss.reset_states()
        for input, labels in data:
          # step_fn(input, labels)
          strategy.run(step_fn, args=(input, labels))
          tl = training_loss.result()
        # break
          # out = 'Loss {:.6f} r: {:.6f}'.format(
          #     round(float(tl), 6),
          #     optimizer._decayed_lr(tf.float32),
          #     )
          # print(out)
          if i > 700 and min_tr_loss > tl:
            # min_tr_loss = tl
            eval_loss.reset_states()
            for input, labels in self.dataset.validation_data: 
              # eval_acc_step_fn(input, labels)
              strategy.run(eval_step_fn, args=(input, labels))

            evaluation_loss = float(eval_loss.result())
            r = optimizer._decayed_lr(tf.float32)
            out = 'Date\t{}\tEpoch\t{}\tLoss\t{:.6f}\tVal loss\t{:.8f}\ttime:\t{:.0f}\tlr:\t{:.6f}\n'.format(
              str(self.train_range.start),
              i + nsteps,
              round(float(tl), 6),
              round(evaluation_loss, 6),
              # round(eval_acc, 6),
              time.time() - start,
              r,
              )
            print(out)
            # logger.warning(out)
            if evaluation_loss < min_eval_loss:
              self.model.save_weights(self.output_path + 'saved.h5')
              with open(self.output_path + '../file.log', 'a') as f:
                f.write(out)
              min_eval_loss = evaluation_loss
              # if evaluation_loss < 0.000142:
              #   return
            if i > 1100:
              return
            pass

  def convert_text_to_indices(self, sentence):
    sentence = self.dataset.bpe.process_line(sentence)
    enc_input = [self.dataset.get_word_index(word) for word in sentence.split()] #+ [self.dataset.PAD_token] * 4
    # dec_input = [self.dataset.SOS_token] #+ enc_input
    enc_input = tf.convert_to_tensor(enc_input)
    input_tensor = tf.reshape(enc_input,(1,-1))
    return input_tensor

  def predict(self, model, inp):
    output = model(inp)
    cat = tf.argmax(output, -1).numpy().squeeze((-1))[0]
    account = self.dataset.index2word[cat]
    return account

from transformers import TFDistilBertForSequenceClassification, DistilBertConfig, DistilBertTokenizerFast

class stock_distilled_bert(stock):
  pretrained_model_name = 'distilbert-base-uncased-finetuned-sst-2-english'
  def init_model(self):
    config = DistilBertConfig.from_pretrained(self.pretrained_model_name)
    config.num_labels = 1
    # config.vocab_size = len(self.dataset.tok.vocab)
    # config.hidden_size = 512
    # config.intermediate_size = 2048
    # config.num_attention_heads = 8
    self.model = TFDistilBertForSequenceClassification(config)
    self.model.layers[0].trainable = False

  def init_dataset(self, dev='CPU', nsteps=10):
    # train_range, test_range = self.set_date_ranges()
    self.dataset = stock_data(
      self, nsteps, #self.train_range, self.test_range,
      tokenizer=DistilBertTokenizerFast.from_pretrained(self.pretrained_model_name),
      device=dev, #enc='cp1251', min_frequency=43, 
      )

  def load_embeddings(self):
    pretrained = TFDistilBertForSequenceClassification.from_pretrained(self.pretrained_model_name)
    self.model.layers[0].set_weights(pretrained.layers[0].get_weights())
    # self.model.summary()
    
from transformers import TFConvBertForSequenceClassification, ConvBertConfig, ConvBertTokenizerFast

class stock_convbert(stock):
  def init_model(self):
    config = ConvBertConfig()
    config.num_labels = 1
    config.vocab_size = len(self.dataset.tok.vocab)
    # config.hidden_size = 1024
    # config.embedding_size = 1024
    # config.intermediate_size = 4096
    # config.num_attention_heads = 16
    # config.num_hidden_layers = 24
    self.model = TFConvBertForSequenceClassification(config)

  def init_dataset(self, dev='CPU', nsteps=10):
    super().init_dataset(dev, nsteps)
    if dev != 'CPU':
      self.dataset.set_batch_sizes_based_on(64)

class stock_convbert_from_pretrained(stock_convbert):
  pretrained_model_name = 'YituTech/conv-bert-base'
  def init_model(self):
    config = ConvBertConfig.from_pretrained(self.pretrained_model_name)
    config.num_labels = 1
    self.model = TFConvBertForSequenceClassification(config)
    self.model.layers[0].trainable = False

  def init_dataset(self, dev='CPU', nsteps=10):
    # train_range, test_range = self.set_date_ranges()
    self.dataset = stock_data(
      self, nsteps, #self.train_range, self.test_range,
      tokenizer=ConvBertTokenizerFast.from_pretrained(self.pretrained_model_name),
      device=dev, #enc='cp1251', min_frequency=43, 
      )

  def load_embeddings(self):
    pretrained = TFConvBertForSequenceClassification.from_pretrained(self.pretrained_model_name)
    self.model.layers[0].set_weights(pretrained.layers[0].get_weights())
    # self.model.summary()
    
from transformers import TFLongformerForSequenceClassification, LongformerConfig

class stock_longformer(stock):
  def init_model(self):
    config = LongformerConfig()
    config.num_labels = 1
    config.vocab_size = len(self.dataset.tok.vocab)
    self.model = TFLongformerForSequenceClassification(config)

  # def init_dataset(self, dev='CPU', nsteps=10):
  #   train_range = time_range(datetime(2010, 1, 1), datetime(2021, 1, 1))
  #   test_range = time_range(datetime(2021, 1, 1), datetime(2021, 1, 10))
  #   self.dataset = stock_data(
  #     self.path_to_dataset, nsteps, train_range, test_range,
  #     device=dev, enc='cp1251', min_frequency=43, 
  #     )
    # if dev != 'CPU':
    #   self.dataset.set_batch_sizes_based_on(64)

def group_companies():
  md = market_data()
  # c = md.returns.corr()
  # c = md.returns.A.corr(md.returns.K)
  d = md.returns.stack().std()
  m = md.returns.stack().mean()
  # M = len(md.returns.index)
  # N = len(md.returns.columns)
  # import numpy as np
  # df_rand = pd.DataFrame(np.random.normal(m, d, (M, N)), columns=md.returns.columns, index=md.returns.index)
  # md.returns[pd.isnull(md.returns)] = df_rand[pd.isnull(md.returns)]
  md.returns.dropna(axis=0, how='all', inplace=True)
  md.returns.dropna(axis=1, inplace=True)
  data = md.returns.to_numpy().T
  nclusters = 10
  from sklearn.cluster import KMeans
  kmeans = KMeans(n_clusters=nclusters).fit(data)

  comp_article_count_path = 'data/comp_article_count.pkl'
  if os.path.exists(comp_article_count_path):
    with open(comp_article_count_path, 'rb') as f:
      comp_article_count = pickle.load(f)
  else:
    def comps_it():
      from stock_data import stock_data_it
      stock_it = stock_data_it('../data/', "../download_from_CC/result")
      for article in stock_it.sentences_it(time_range(), roundup=False, verbose=1):
        yield article[3]

    from collections import Counter
    comp_article_count = Counter(comps_it())
    with open(comp_article_count_path, 'wb') as f:
      pickle.dump(comp_article_count, f)

  group_count = [0] * nclusters
  comps_grouped = [[] for _ in range(nclusters)]
  for comp, g in zip(md.returns.columns, kmeans.labels_):
    group_count[g] += comp_article_count[comp]
    comps_grouped[g] += [comp]

    # with open('output.txt', 'w') as f:
    #   for k, v in c.items():
    #     f.write(k + '\t' + str(v) + '\n')

  sorted_groups = [(g, s) for g, s in enumerate(group_count)]
  sorted_groups.sort(key=lambda tup: tup[1], reverse=True)
  num_articles = sum(group_count)
  max_len = num_articles / len(group_count)

  for i, g in enumerate(sorted_groups):
    if g[1] > max_len:
      for comp in comps_grouped[g[0]]:
        for gr in range(i + 1, len(sorted_groups)):
            
          pass

def train_sample():
  path_to_working_days = ''
  if 'kaggle' in globals():
    path_to_working_days = '/kaggle/input/working-days/'
  df = pd.read_excel(
    path_to_working_days + 'analysis_net_of_index_best_eval_loss.xlsx', 
    sheet_name='work days sample', 
    header=0,
    engine='openpyxl'
    )
  # inds = [35, 36, 37, 39, 40, 41, 42, 43, 46, 47, 49, 55]
  # dates = [df.eval_start.dt.date[i] for i in inds]
  # for eval_start in dates:
  for eval_start in df.eval_start.dt.date[:60]:
    wdi_eval = df.index[df.eval_start_sorted.dt.date == eval_start]
    try:
      eval_end, train_start, train_end, test_start, test_end = \
        df.eval_start_sorted.dt.date.iloc[wdi_eval + 1].values[0], \
        df.eval_start_sorted.dt.date.iloc[wdi_eval + 2].values[0], \
        df.eval_start_sorted.dt.date.iloc[wdi_eval + 24].values[0], \
        df.eval_start_sorted.dt.date.iloc[wdi_eval + 25].values[0], \
        df.eval_start_sorted.dt.date.iloc[wdi_eval + 26].values[0]
    except:
      continue
    print(eval_start, eval_end, train_start, train_end, test_start, test_end)
    # s = stock_convbert(time_range(eval_start, eval_end), time_range(train_start, train_end), cloud)
    # s.init_dataset()
    # s.dataset.prepare_for_training()
    # s.train()
    stock_convbert(time_range(test_start, test_end), time_range(train_start, train_end), cloud).test()
    pass

def daily_prepare_for_training():
  s = get_daily_trainer()
  s.init_dataset()
  s.dataset.prepare_for_training()
  pass

def daily_train():
  s = get_daily_trainer()
  s.train()
  pass

def get_daily_trainer():
    df = load_work_days()
    eval_start, eval_end, train_start, train_end = \
    df.DATE.dt.date.iloc[-25], \
    df.DATE.dt.date.iloc[-24], \
    df.DATE.dt.date.iloc[-23], \
    df.DATE.dt.date.iloc[-1]
    s = stock_convbert(time_range(eval_start, eval_end), time_range(train_start, train_end), cloud)
    return s

def load_work_days():
    path_to_working_days = '../data/'
    if 'kaggle' in globals():
      path_to_working_days = '/kaggle/input/market-data/'
    df = pd.read_excel(
      path_to_working_days + 'sp_500.xlsx', 
      header=0,
      engine='openpyxl'
    )
    return df

def daily_test():
  df = load_work_days()
  test_start, test_end, train_start, train_end = \
  df.DATE.dt.date.iloc[-1], \
  None, \
  df.DATE.dt.date.iloc[-23], \
  df.DATE.dt.date.iloc[-1]
  stock_convbert(time_range(test_start, test_end), time_range(train_start, train_end), cloud).test()

# tf.random.set_seed(0)
# daily_prepare_for_training()
# daily_train()
# daily_test()
train_sample()
# import subprocess
# subprocess.run(
#   ['kaggle', 'dataset', 'version', '-p', '' 
#   '--recursive'], check=True)
pass
