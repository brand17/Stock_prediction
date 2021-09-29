import pickle
import numpy as np
import tensorflow as tf
import tqdm, os, re
import pandas as pd
from pathlib import Path
from datetime import timedelta

class stock_data_it():
  def __init__(self, path_to_market_data, path_to_dataset):
    self.path_to_dataset = path_to_dataset
    from market_data import market_data
    self.md = market_data(
      path_to_market_data + 'market_data.xlsx', 
      path_to_market_data + 'sp500.xlsx'
      )

  def files_it(self, verbose=0):
    # pathlist = Path(self.path_to_dataset).glob('*.parquet')
    pathlist = list(self.get_path_list())
    # from utils import GetFolderSize
    #errorRegex=re.compile(r'\s[^авикосуя]\s[^авикосуя]\s|(\s|\A)[^аеёиоуыэюя][^аеёиоуыэюя]+(\s|\Z)')
    # with tqdm.tqdm(total=GetFolderSize(p)) as pbar:
    # from random import shuffle
    # shuffle(pathlist)
    for path in sorted(pathlist):
      if verbose == 1:
        print(path)
      yield path

  def get_path_list(self):
    pathlist = Path(self.path_to_dataset).glob('*.parquet')
    return pathlist

  def articles_it(self, r=None):
    for path in self.files_it():
      for a in self.articles_it_in_file(path, r):
        yield a

  def dataframe_from_file(self, path, r):
    table = pd.read_parquet(path)
    table = table[(table['return'].notnull())]
    if r is not None:
      if r.start is not None:
        table = table[table.publ_time.dt.date >= r.start]
      if r.end is not None:
        table = table[table.publ_time.dt.date < r.end]
    if len(table.index) > 0:
      publ_date = (table.publ_time - timedelta(hours=40)).dt.date
      li = self.md.returns.index.get_indexer(publ_date)
      ri = self.md.returns.columns.get_indexer(table.comp_short)
      table['return'] = self.md.returns.values[li, ri]
    return table

  def articles_it_in_dataframe(self, table):
    for text, s, publ_time, comp_short in zip(table.content, table['return'], table.publ_time, table.comp_short):
      text = self.string_from_file(text)
      yield text, s, publ_time, comp_short

  def articles_it_in_file(self, path, r=None):
    table = self.dataframe_from_file(path, r)
    for article in self.articles_it_in_dataframe(table):
      yield article

  def sentences_it(self, r, roundup=True, verbose=0):
    while True:
      for path in self.files_it(verbose):
        for articleID, article in enumerate(self.articles_it_in_file(path, r)):
          for sent in article[0]:
            yield (sent,) + article[1:] + (articleID,)
      if not roundup:
        break

  def sentences_it_text_only(self, r, verbose=0):
    for sent in self.sentences_it(r, roundup=False, verbose=verbose):
      yield sent[0]

  def string_from_file(self, text):
    # text = re.sub(r"(?<=\W)[А-Я]\.", '', text)  # remove initials
    text = re.sub(r'\n\s*(?=\n)', r"", text)  # remove empty strings
    # text = re.sub(r'(\w)-\s*\n\s*(\w)', r"\1\2", text)  # remove hyphens
    # text = re.sub(r'\.\s*\n(\w)', r".\n \1", text) # avoid merging paragraphs if ended by dot
    # text = re.sub(r'\n(\w)', r" \1", text)  # merge paragraphs
    text = re.sub(r"[0-9].[0-9]", "0", text)  # remove decimal
    text = re.sub(r"[0-9]+", "0", text)  # merge digits
    # text = re.sub(r"[a-z]+", 'a', text)  # merge english
    # text = re.sub(r"\.\.+", '.', text)  # merge dots
    # text = re.sub(r'\.\s+([A-Z])', r"\n\1", text)  # split by sentences
    # text = re.sub(r'ё', r"е", text)  # replace ё
    # text = re.sub(r" +", r" ", text)  # merge spaces
    # text = re.sub(r'\s*\n\s*', r"\n", text)  # remove spaces in the beginning
    # text = re.sub(r"^\s+", r"", text)  # merge spaces
    text = text.lower()
    strings = text.split('\n')
    return strings

class stock_data(stock_data_it):
  def __init__(self, trainer, nsteps, tokenizer=None,
               device='CPU'):
    # print('initializing dataset ' + file_name)
    self.lengths = [8, 16, 24, 32, 40, 48, 64, 80, 128, 256, 512]
    self.nsteps = nsteps
    self.trainer = trainer
    # self.path_to_dataset = trainer.path_to_dataset
    # self.path_to_validation = trainer.path_to_validation
    # self.enc = enc
    # self.min_frequency = min_frequency
    self.device = device
    # self.train_range = trainer.train_range
    # self.test_range = trainer.test_range
    # self.model_name = model_name
    # if os.path.exists(PATH + 'comps.pkl'):
    #   self.comps = word2ind(PATH + 'comps.pkl')
    # else:
    #   self.comps = word2ind()

    from market_data import market_data
    self.md = market_data(
      trainer.path_to_market_data + 'market_data.xlsx', 
      trainer.path_to_market_data + 'sp500.xlsx'
      )

    if tokenizer is None:
      self.tok = self.custom_tokenizer()
    else:
      self.tok = tokenizer

    self.init_buckets()
    
    if self.device == 'CPU':
      print('dataset for CPU !!!')
      self.num_validation_samples = 500
      self.batch_sizes = {l: 8 for l in self.lengths}
    else:
      self.num_validation_samples = 5000
      self.set_batch_sizes()

  def custom_tokenizer(self):
    from transformers import PreTrainedTokenizerFast
    from tokenizers import Tokenizer, trainers
    from tokenizers.models import BPE
    from tokenizers.pre_tokenizers import Whitespace
    # from transformers.models.auto.tokenization_auto import tokenizer_class_from_name
    file_name = self.trainer.path_to_validation + 'tokenizer.json'
    if not os.path.exists(file_name):
      assert self.device == 'CPU', 'No tokenizer found !!! Train on CPU'
      tokenizer = Tokenizer(BPE()) 
      tokenizer.pre_tokenizer = Whitespace()
      from tokenizers.processors import BertProcessing
      tokenizer.post_processor = BertProcessing(sep=("[EOS]", 2), cls=("[BOS]", 0))
      trainer = trainers.BpeTrainer(special_tokens=["[BOS]", "[PAD]", "[EOS]", "[UNK]", '[MASK]'])
      tokenizer.train_from_iterator(self.sentences_it_text_only(self.trainer.train_range, verbose=1), trainer)
      tokenizer.save(file_name)

    tokenizer = PreTrainedTokenizerFast(
      tokenizer_file=file_name, 
      bos_token='[BOS]', pad_token='[PAD]', eos_token='[EOS]',
      unk_token='[UNK]', cls_token='[BOS]', sep_token='[EOS]', mask_token='[MASK]'
      )
    # self.save_token_frequencies(tokenizer)
    return tokenizer

  # def save_token_frequencies(self, tokenizer):
  #   from collections import Counter
  #   counter = Counter()
  #   from time import time
  #   start = time()
  #   for s in self.sentences_it_text_only(self.test_range, verbose=1):
  #     tokens = tokenizer(s)['input_ids']
  #     counter.update(tokens)

  #   with open('output.txt', 'w') as f:
  #     for w, c in counter.items():
  #       f.write(str(w) + '\t' + str(c) + '\n')

  #   print(time() - start)

  def get_path_list(self):
    pathlist = Path(self.trainer.path_to_dataset).glob('part-00294*.parquet')
    return pathlist

  def prepare_for_training(self):
    self.validation_data = self.get_validation_sentences(self.trainer.test_range)

  # def input_from_text(self, sentence):
  #   inp = self.tok(sentence, return_token_type_ids=False)
  #   return inp
    
  def process_sentence(self, sent):
    # inp = self.input_from_text(sent[0])
    inp = self.tok(sent[0], return_token_type_ids=False)
    l = len(inp['input_ids'])
    length = self.get_bucket_length(l)
    if length is None:
      return None
    self.buckets[length]['input_ids'] += [inp['input_ids'] + [self.tok.pad_token_id] * (length - l)]
    self.buckets[length]['attention_mask'] += [inp['attention_mask'] + [0] * (length - l)]
    self.buckets[length]['scores'] += [sent[1]]
    self.buckets[length]['publ_time'] += [sent[2]]
    self.buckets[length]['comp_short'] += [sent[3]]
    self.buckets[length]['articleID'] += [sent[4]]
    return length

  def get_bucket_length(self, l):
    for length in self.buckets:
      if l <= length:
        return length

  def init_buckets(self):
    self.buckets = {}
    [self.init_bucket(k) for k in self.lengths]

  def init_bucket(self, k):
    self.buckets[k] = {
      'input_ids': [], 'attention_mask': [], 'publ_time': [],
      'scores': [], 'comp_short': [], 'articleID': []
      }
    
  def load_batch(self, r, k=None, batch_size=None, roundup=True, verbose=0):
    for sent in self.sentences_it(r, roundup, verbose):
      length = self.process_sentence(sent)
      if length is not None:
        if k is not None and length != k:
          continue
        if batch_size == None:
          s = self.batch_sizes[length] * self.nsteps
        else:
          s = batch_size
        if len(self.buckets[length]['input_ids']) == s:
          yield self.batch_from_bucket(length)
    for l in self.lengths:
      if len(self.buckets[l]['input_ids']) > 0:
        yield self.batch_from_bucket(l)

  def batch_from_bucket(self, length):
    res = self.buckets[length]
    self.init_bucket(length)
    return np.array(res['input_ids'], dtype=np.int32), \
      np.array(res['scores'], dtype=np.float32), \
      res['publ_time'], \
      res['comp_short'], \
      res['articleID']

  def get_validation_sentences(self, r):
    file_name = self.trainer.path_to_validation + 'validation_data.pkl'
    if not os.path.isfile(file_name):
      assert self.device == 'CPU', 'No validation file found !!! Run on CPU to create (is slow on colab TPU)'
      pathlist = list(self.get_path_list()) # list(Path(self.path_to_dataset).glob('*.parquet'))
      file_probabilities = [v.stat().st_size for v in pathlist]
      import random
      sample = random.choices(pathlist, file_probabilities, k=5000)
      from collections import Counter
      sample = Counter(sample)
      import pandas as pd
      res = dict(self.buckets[8])
      print('sampling sentences for evaluation dataset...')
      with tqdm.tqdm(total=len(sample)) as pbar:
        for k, v in sample.items():
          table = self.dataframe_from_file(k, r)
          if len(table) > 0:
            table = pd.DataFrame.sample(table, v, replace=True)
            articles = list(self.articles_it_in_dataframe(table))
            for article in articles:
              text = random.sample(article[0], 1)
              inp = self.tok(text[0], return_token_type_ids=False)
              # inp = self.input_from_text(text[0])
              l = len(inp['input_ids'])
              if l > 512:
                continue
              res['input_ids'] += [inp['input_ids'] + [self.tok.pad_token_id] * (512 - l)]
              res['scores'] += [article[1]]
              res['publ_time'] += [article[2]]
              res['comp_short'] += [article[3]]
              res['articleID'] += [0]
          pbar.update(1)

      sentences = np.array(res['input_ids'], dtype=np.int32), \
        np.array(res['scores'], dtype=np.float32), \
        res['publ_time'], \
        res['comp_short'], \
        res['articleID']
      with open(file_name, 'wb') as f:
        pickle.dump(sentences, f)
    else:
      print('loading evaluation dataset...')
      with open(file_name, 'rb') as f:
        sentences = pickle.load(f)
    if self.device == 'CPU':
      sentences = [s[:self.num_validation_samples] for s in sentences]
    sentences = self.dataset_from_target(sentences[:2], drop_remainder=True)
    # if self.device == 'TPU':
    #   sentences = self.dataset_from_target(sentences[:2], drop_remainder=True)
    # else:
    #   sentences = self.dataset_from_target(sentences[:2], drop_remainder=True)
    return sentences

  def next(self, verbose=0, roundup=True):
    for b in self.load_batch(self.trainer.train_range, verbose=verbose, roundup=roundup):
      sentences = self.dataset_from_target(b[:2], drop_remainder=True)
      yield sentences
  
  def dataset_from_target(self, tar, drop_remainder=False):
    att_mask = tf.cast(tf.math.not_equal(tar[0], self.tok.pad_token_id), tf.int32)
    if len(tar) > 1:
      sentences = tf.data.Dataset.from_tensor_slices(((tar[0], att_mask), tar[1]))
    else:
      sentences = tf.data.Dataset.from_tensor_slices((tar[0], att_mask))
    return self.batch_dataset(sentences, tar[0].shape[1], drop_remainder)

  def batch_dataset(self, sentences, bi, drop_remainder=False):
    return sentences.batch(self.batch_sizes[bi], drop_remainder)

  def set_batch_sizes(self):
    self.set_batch_sizes_based_on(128)

  def set_batch_sizes_based_on(self, mult):
    multiplier = {k: mult * 512 / (k * 128) for k in self.buckets.keys()}
    import math
    multiplier = {k: m if m < 1 else math.floor(m) for k, m in multiplier.items()}
    self.batch_sizes = {k: int(128 * m) for k, m in multiplier.items()}
    if self.device == 'TPU':
      assert mult >= 8

