from paraphrase_data import paraphrase_dataset
import numpy as np
import tensorflow as tf
import tqdm, os, re
# from transformers import MarianTokenizer
# from tokenizers.models import BPE
# from tokenizers.trainers import BpeTrainer
# from tokenizers.pre_tokenizers import Whitespace
from pathlib import Path
import io
import pandas as pd
from utils import check_file
import struct
# import tempfile
import pickle

class stock_data(paraphrase_dataset):
  def initBPE(self):
    from transformers import RobertaTokenizerFast
    self.tok = RobertaTokenizerFast.from_pretrained('roberta-base')
    self.EOS_token = self.tok.eos_token_id
    self.PAD_token = self.tok.pad_token_id
    self.SOS_token = self.tok.pad_token_id
    self.word2index = self.tok.vocab
    l = len(self.word2index)
    self.index2word = [0] * l
    for k, v in self.word2index.items():
      self.index2word[v] = k

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

  def data_it(self, name='', enc='UTF-8'):
    from pathlib import Path
    pathlist = Path("../download_from_CC/result").glob('*.parquet')
    # pathlist = Path("../download_from_CC/result").glob('part-00362*.parquet')
    # from utils import GetFolderSize
    #errorRegex=re.compile(r'\s[^авикосуя]\s[^авикосуя]\s|(\s|\A)[^аеёиоуыэюя][^аеёиоуыэюя]+(\s|\Z)')
    # with tqdm.tqdm(total=GetFolderSize(p)) as pbar:
    for path in sorted(pathlist):
      # with open("log.txt", "a+") as myfile:
      #   print(path, file=myfile)
      print(path)
      sentences = self.readTrainFile(path)
      # read_len = os.path.getsize(path)
      for sent in sentences:
        yield sent
      # pbar.update(read_len)

  def readTrainFile(self, path):
    table = pd.read_parquet(path)
    # table = table[(table.publ_time >= self.min_date) & (table.publ_time < self.max_date)]
    table = table.content
    text = ' '.join(table)
    text = self.string_from_file(text)
    return text

  def load_dataset(self, file_name):
    # remove rare tokens and convert data to int16
    if not os.path.isfile(self.PATH + '8'): # convert indices to tensors
        print("Creating sentences")
        self.split_to_buckets(self.PATH + 'result.txt', self.enc)
        #for k, v in self.buckets:
        #  print(k)
        #  save(v, PATH + str(k) + '.dat')
        #save(self.buckets, PATH + 'sentences.dat')

        # with open(self.PATH + file_name, 'wb') as fp:
        #   pickle.dump(self.buckets, fp)

    else:
      print('Loading sentences...')
      #self.buckets = load(PATH + 'sentences.dat')
      # with open(self.PATH + file_name, 'rb') as fp:
      #   self.buckets = pickle.load(fp)
      self.create_buckets(None)
    
    #self.buckets[8] = np.array([]) # remove 8 bucket - not stable

    if self.device == 'CPU':
      num_samples = 500
      self.batch_sizes = [8 for l in self.lengths]
    else:
      num_samples = 5000
      self.set_batch_sizes()
      # self.batch_sizes = [64*512//(k*128)*128 for k in self.buckets.keys()]
      # self.batch_sizes[-1] = 64
      # self.batch_sizes[6] = 256
    self.validation_data = self.get_validation_sentences(num_samples)
    self.optimize_buckets()
    self.set_probabilities()
    # self.bucket_probabilities = [v.shape[0] for v in self.buckets.values()]

  def split_to_buckets(self, name, enc):
    buckets = self.init_buckets()#{k:[] for k in self.lengths}
    for sent in self.data_it(name, enc=enc):
      self.add_sentence_to_bucket(sent, buckets)
    a = {buckets[k].close() for k in self.lengths}
    self.create_buckets(buckets)

  def init_buckets(self):
    return {k:open(self.PATH + str(k), 'wb') for k in self.lengths}
    # return {k:tempfile.TemporaryFile() for k in self.lengths}
    
  def add_sentence_to_bucket(self, sentence, buckets):
    sentence, length = self.process_sentence(sentence, buckets)
    if sentence is not None:
      s = struct.pack('i'*length, *sentence)
      buckets[length].write(s)

  def create_buckets(self, buckets):
    # for b in buckets.values():
    #   random.shuffle(b)
    self.buckets = {}
    for k in self.lengths:
      f = open(self.PATH + str(k), 'rb')
      if self.device == 'CPU':
        f = f.read(k * 1000 * 4)
      else:
        f = f.read()
      sentence = struct.unpack('i' * (len(f) // 4), f)
      self.buckets[k] = np.array(sentence, dtype=np.int32).reshape((-1, k))
    # self.buckets = {k:np.array(sentence, dtype=np.int32) for k, sentence in buckets.items()}

  def tokenize(self, sentence):
    tokens = self.tok.tokenize(sentence)
    return self.tok.convert_tokens_to_ids(tokens)
      
  # def process_sentence(self, sentence, buckets):
  #   l = self.get_bucket(sentence) #len(sentence)
  #   length = self.get_bucket_length(l, buckets)
  #   sentence = sentence + [self.EOS_token] + [self.PAD_token] * (length - l - 1)
  #   return sentence, length

  def input_from_target(self, tar):
    #tar = np.random.randint(0, 20000, [26880, 24], dtype=np.int32)
    enc_input = np.copy(tar)
    # dec_input = np.roll(tar, 1, -1)
    sos = np.full((enc_input.shape[0], 1), self.SOS_token)
    #enc_input = np.concatenate((sos, enc_input), -1)
    dec_input = np.concatenate((sos, tar[:,:-1]), -1)
    np.place(dec_input, dec_input==self.EOS_token, [self.PAD_token]) # remove this line because pretrained BERT EOS is different from PAD
    sentences = ((enc_input, dec_input), tar)
    bi = self.lengths.index(tf.shape(enc_input)[1])
    return sentences, bi

