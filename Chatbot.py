# !/usr/bin/python 
# -*-coding:utf-8 -*- 
import sys, os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
# from scipy.misc import imread, imsave, imresize
from PIL import Image
from scipy import signal
import time, datetime, random
import re, requests, json
import numpy as np
import tensorflow as tf
import asyncio
import telepot
import telebot
from telepot.aio.loop import MessageLoop
from telepot.aio.delegate import per_chat_id, create_open, pave_event_space, include_callback_query_chat_id
from telepot.namedtuple import InlineQueryResultArticle, InputTextMessageContent
from telepot.namedtuple import InlineKeyboardMarkup, InlineKeyboardButton, ReplyKeyboardMarkup, ReplyKeyboardRemove, KeyboardButton
from keras.models import Model, load_model
from keras import backend as K
from keras.engine.topology import Layer
from keras.models import model_from_json
from vis.visualization import visualize_saliency, visualize_cam, visualize_activation


#
from chatterbot import ChatBot

bot = telebot.TeleBot

def bot_conversation(message):

    chatbot = ChatBot("Hao", trainer = "chatterbot.trainers.ChatterBotCorpusTrainer")
    request = chatbot.get_response(message)
    request = str(request)

    message = open("request.txt","w")
    message = write(request)
    message.close()

@bot.message_handler(commands = ["help","start"])
def send_message(message):
    bot.reply_to(message,".")


@bot.message_handler(func=lambda message:True)
def message(message):
    bot_conversation(message.text)
    request = open("request.txt", 'r')    
    request = request.read()
    bot.reply_to(message,request)


mDict = json.load(open('./object_dict.json')) # This very important to maintain 
model_path = "./model/ResNet50_try_man/ResNet50_try_man.h5"


class Manifold_loss(Layer):

    def __init__(self, alpha=1., **kwargs):
        super(Manifold_loss, self).__init__(**kwargs)
        #self.supports_masking = True
        self.alpha = K.cast_to_floatx(alpha)

    def call(self, inputs):
        return manifold_loss(inputs, manifold_ratio=self.alpha)

    def get_config(self):
        config = {'alpha': float(self.alpha)}
        base_config = super(Manifold_loss, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    # @tf_utils.shape_type_conversion
    def compute_output_shape(self, input_shape):
        return input_shape


def manifold_loss(x, manifold_ratio):

    x = tf.cast(x, tf.complex128)
    x = tf.fft(x)
    temp_x = 2*tf.real(x)/(1+tf.real(x)*tf.real(x)+tf.imag(x)*tf.imag(x))
    temp_y = 2*tf.real(x)/(1+tf.real(x)*tf.real(x)+tf.imag(x)*tf.imag(x))

    Z_Q_1 = 2*tf.sqrt((1+temp_x*temp_x+temp_y*temp_y)*tf.sqrt(temp_x*temp_x+temp_y*temp_y)-(temp_x*temp_x+temp_y*temp_y))/(1+temp_x*temp_x+temp_y*temp_y)

    point_x  = 180*tf.cos( 2 * tf.atan(temp_y/(tf.sqrt(temp_y**2 + temp_x**2) + temp_x)) )
    point_y = 180*tf.sin(tf.asin(Z_Q_1)/np.pi)

    point_x = (1+tf.cos(point_x)) * tf.cos(point_y)
    point_y = (1+tf.cos(point_x)) * tf.sin(point_x)

    N = tf.sqrt(tf.reduce_sum(tf.pow(point_x-point_y,2),keepdims=True))
    N = tf.multiply(N,manifold_ratio)
    N = tf.cast(N, tf.float32)
        
    #x[1]=N
    return N #real output

model = load_model(model_path,custom_objects={"Manifold_loss":Manifold_loss})

class User:
    def __init__(self, chatid):        
        self.chat_id = chatid

def getUser(chat_id):
    for user in users:
        if user.chat_id == chat_id:
            return user
    return None

def formatMsg(msg):
    info = 'I think it is..\n'
    reply =  info + '👉🏼' + str(msg)
    return reply

def formatTop5Msg(result):
    reply = '[Top5]\n'
    for r in result:
       reply += r[0] + ' : ' + str(r[1]) + '\n'
    return reply

def boxing(img, label):
 
    layer_idx = [idx for idx, layer in enumerate(model.layers) if layer.name == "dense_2"][0]
    heatmap = visualize_saliency(model, layer_idx, np.expand_dims(label, axis=0), img)
    k_size = 28
    k = np.ones((k_size,k_size))/k_size
    heatmap = signal.convolve2d(heatmap[:,:], k, boundary='wrap', mode='same')/k.sum()
    threshold = heatmap.max() * 0.3
    maxTop = maxLeft = 999999999
    maxRight = maxBottom = -1
    for h in range(224):
        for w in range(224):
            # print(h,w)
            if heatmap[h][w] > threshold:
                if h < maxTop: maxTop = h
                if h > maxBottom: maxBottom = h
                if w < maxLeft: maxLeft = w
                if w > maxRight: maxRight = w


    maxTop = int(maxTop/3)
    maxBottom = int(maxBottom/3)
    maxLeft = int(maxLeft/3)
    maxRight = int(maxRight/3)
    img = img.copy()
    for h in range(224):
        for w in range(224):
            if (int(h/3) == maxTop and int(w/3) in range(maxLeft, maxRight)) or (int(h/3) == maxBottom and int(w/3) in range(maxLeft, maxRight)) or (int(w/3) == maxRight and int(h/3) in range(maxTop, maxBottom))  or (int(w/3) == maxLeft and int(h/3) in range(maxTop, maxBottom)):
                img[h][w][0] = img[h][w][1] = 255
                img[h][w][2] = 0

    return img

users = [] 
service_keyboard = ReplyKeyboardMarkup(
                            keyboard=[
                                [KeyboardButton(text="Random"),KeyboardButton(text="Help")], 
                            ]
                        ) 
def getKeybyVal(mDict,idx):
    return list(mDict.keys())[list(mDict.values()).index(idx)]


def getSampleImages():
    imgs = [f for f in os.listdir('sample-img') if os.path.isfile(os.path.join('sample-img', f))]
    return imgs

class PRBot(telepot.aio.helper.ChatHandler):

    def __init__(self, *args, **kwargs):
        super(PRBot, self).__init__(*args, **kwargs)

    async def on_chat_message(self, msg):      
        content_type, chat_type, chat_id = telepot.glance(msg)

        if content_type == 'photo':
            # download image and predict
            await bot.download_file(msg['photo'][-1]['file_id'], 'img/tmpImg.png')
            img = Image.open('img/tmpImg.png')
            img = img.resize((224,224), Image.BILINEAR)
            img = np.asarray(img)
            prob = model.predict(np.expand_dims(img, axis=0))
            
            # get top 5
            sorted_idx = list(np.argsort(prob[0]))
            sorted_idx = sorted_idx[::-1]
            top_result = getKeybyVal(mDict, sorted_idx[0])
            top5_idx = sorted_idx[0:5]
            top5_result = []
            for idx in top5_idx:
                top5_result.append([getKeybyVal(mDict, idx),  '{:.4%}'.format(prob[0][idx])])

            # boxing
            bbox_img = boxing(img,sorted_idx[0])
            save_img_name = '.' + str(chat_id) + '_bbox_img.png'
            result = Image.fromarray(bbox_img)
            result.save(os.path.join('img', save_img_name))
            # send result
            #await self.sender.sendPhoto(open(os.path.join('img', save_img_name), 'rb')) 
            await self.sender.sendMessage(formatMsg(top_result))

            os.remove(os.path.join('img', save_img_name))
            os.remove(os.path.join('img', 'tmpImg.png'))
            return
        elif content_type == 'text':
            if(getUser(chat_id) is None):
                print("new user", chat_id)
                user = User(chat_id)
                users.append(user)

            msg = msg['text']
            print(chat_id, msg) 

            if msg == '/start':
                await self.sender.sendMessage( "🍀 Upload photo of a plant to classify. ", reply_markup=service_keyboard)
            elif msg == 'Help' or msg == '/help':
                await self.sender.sendMessage( "👉🏼 Use telegram upload photo", reply_markup=service_keyboard)
            elif msg == 'Random':
                filename = random.choice(sampleImgs)
                img = Image.open( os.path.join('sample-img', filename))
                img = img.resize((224,224), Image.BILINEAR)
                img = np.asarray(img)
                label = int(filename[:-4])
                result = getKeybyVal(mDict, label)
                bbox_img = boxing(img,(label))
                save_img = Image.fromarray(bbox_img)
                save_img.save(os.path.join('sample-img', 'bbox_img.png'))
                await self.sender.sendPhoto(open('sample-img/bbox_img.png', 'rb')) 
                await self.sender.sendMessage( formatMsg(result), reply_markup=service_keyboard)
                os.remove(os.path.join('sample-img', 'bbox_img.png'))
        return

           
TOKEN = 'ChatBot' #tk
sampleImgs = getSampleImages()
bot = telepot.aio.DelegatorBot(TOKEN, [
    include_callback_query_chat_id(
        pave_event_space())(
        per_chat_id(), create_open, PRBot, timeout= 120),
])


loop = asyncio.get_event_loop()
loop.create_task(MessageLoop(bot).run_forever())
#print('Listening ...')
loop.run_forever()

