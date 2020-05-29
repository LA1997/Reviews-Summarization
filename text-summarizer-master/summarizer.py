from a import text_cleaner
from a import text_splitter
import pandas as pd
import os
import pickle
import numpy as np  
import pandas as pd 
import re           
from bs4 import BeautifulSoup 
from keras.preprocessing.text import Tokenizer 
from keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords   
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense, Concatenate, TimeDistributed, Bidirectional
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
import warnings
import time
pd.set_option("display.max_colwidth", 200)
warnings.filterwarnings("ignore")
import subprocess
# from scrapy.crawler import CrawlerProcess
# from AmazonReviews.AmazonReviews.spiders import getreviews



def seq2summary(input_seq):
    newString=''
    for i in input_seq:
      if((i!=0 and i!=target_word_index['start']) and i!=target_word_index['end']):
        newString=newString+reverse_target_word_index[i]+' '
    return newString

def seq2text(input_seq):
    newString=''
    for i in input_seq:
      if(i!=0):
        newString=newString+reverse_source_word_index[i]+' '
    return newString

def getKeywords(text):
    return text[len(text)-1]
def encoderdecodermodels():
    
    modelc = open("summarizer.pickle", "rb")
    model = pickle.load(modelc)
    modelc.close()
    return
    encoder_model = Model(inputs=model[0],outputs=[model[1], model[2], model[3]])
    latent_dim = model[4][0]
    decoder_state_input_h = Input(shape=(latent_dim,))
    decoder_state_input_c = Input(shape=(latent_dim,))
    decoder_hidden_state_input = Input(shape=(max_len_text,latent_dim))

    dec_emb2= dec_emb_layer(decoder_inputs)

    decoder_outputs2, state_h2, state_c2 = decoder_lstm(dec_emb2, initial_state=[decoder_state_input_h, decoder_state_input_c])

    attn_out_inf, attn_states_inf = attn_layer([decoder_hidden_state_input, decoder_outputs2])
    decoder_inf_concat = Concatenate(axis=-1, name='concat')([decoder_outputs2, attn_out_inf])

    decoder_outputs2 = decoder_dense(decoder_inf_concat)

    decoder_model = Model(
    [decoder_inputs] + [decoder_hidden_state_input,decoder_state_input_h, decoder_state_input_c],
    [decoder_outputs2] + [state_h2, state_c2])
   
def decode_sequence(input_seq):
    e_out, e_h, e_c = encoder_model.predict(input_seq)
    target_seq = np.zeros((1,1))

    target_seq[0, 0] = target_word_index['start']

    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + [e_out, e_h, e_c])
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_token = reverse_target_word_index[sampled_token_index]

        if(sampled_token!='end'):
            decoded_sentence += ' '+sampled_token

            if (sampled_token == 'end' or len(decoded_sentence.split()) >= (max_len_summary-1)):
                stop_condition = True

        target_seq = np.zeros((1,1))
        target_seq[0, 0] = sampled_token_index
        e_h, e_c = h, c

    return decoded_sentence

def text_clean(text):
    newString = text.lower()
    newString = BeautifulSoup(newString, "lxml").text
    newString = re.sub(r'\([^)]*\)', '', newString)
    newString = re.sub('"','', newString)
    newString = ' '.join([contraction_mapping[t] if t in contraction_mapping else t for t in newString.split(" ")])    
    newString = re.sub(r"'s\b","",newString)
    newString = re.sub("[^a-zA-Z]", " ", newString) 
    tokens = [w for w in newString.split() if not w in stop_words]
    long_words=[]
    for i in tokens:
        if len(i)>=3:                  #removing short word
            long_words.append(i)   
    return (". ".join(long_words)).strip()

def getSumByURL(u):
    os.chdir('/home/loki/Documents/Summer Training 4thyr/text-summarizer-master/AmazonReviews')
    print('\n\n\n\n\n\n\n\n')
    os.system('pwd')
    dfa = pd.DataFrame({'url':u,'asin':''},columns=['url','asin'],index=[0])
    dfa.to_csv('Op.csv')
    subprocess.run(['scrapy','crawl','getreviews1'])
    time.sleep(20)
    os.chdir('/home/loki/Documents/Summer Training 4thyr/text-summarizer-master')
    
    
def getSumByASIN(u):
    os.chdir('/home/loki/Documents/Summer Training 4thyr/text-summarizer-master/AmazonReviews')
    dfa = pd.DataFrame({'url':'','asin':u},columns=['url','asin'],index=[0])
    dfa.to_csv('Op.csv')
    subprocess.run(['scrapy','crawl','getreviews2'])
    #time.sleep(20)
    os.chdir('/home/loki/Documents/Summer Training 4thyr/text-summarizer-master')
    

def getSum(u):
    if(u[0]!=''):
        getSumByURL(u[0])
        df = pd.read_csv('ReviewsScraped.csv')
        df.dropna(axis=0,inplace=True)
        text = '; '.join(list(df['Reviews'])) 
    
    elif(u[1]!=''):
        getSumByASIN(u[1])
        df = pd.read_csv('ReviewsScraped.csv')
        df.dropna(axis=0,inplace=True)
        text = '; '.join(list(df['Reviews'])) 
    
    elif(u[2]!=''):
        text = u[2] 
        
    text_cleaner(text)
    encoderdecodermodels()
    textseq = text.split(' ')
    print(textseq)
    res = [i for i in range(len(textseq)) if textseq[i].lower() =='not']
    print(res)
    summary = text_splitter(text)
    print(summary)
    keywords = getKeywords(summary)
    summary = summary[0]
    #summary = ' '.join(list(set(summary[len(summary)-2].replace('\n','').replace('.','').split(' '))))
    for i in range(len(res)):
        if(res[i]+2<len(textseq)):
            summary += '; '+textseq[res[i]]+' '+textseq[res[i]+1]+' '+textseq[res[i]+2]
            keywords+= ' '+(textseq[res[i]]+' '+textseq[res[i]+1]+' '+textseq[res[i]+2])
        else: 
            summary += '; '+textseq[res[i]]+' '+textseq[res[i]+1]
            keywords+= ' '+(textseq[res[i]]+' '+textseq[res[i]+1])

        
    return (summary.split(' '),keywords)


