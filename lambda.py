import json
import keras
from PIL import Image
import boto3
import io

import base64
import numpy as np
from keras.preprocessing.image import img_to_array, load_img
my_bucket='workattr'
strkey='weights-FC37-MobileNetV2-0.92.hdf5'
cl=boto3.client('s3')
cl.download_file(my_bucket,strkey,'/tmp/weights-FC37-MobileNetV2-0.92.hdf5')
model = keras.models.load_model(f"/tmp/weights-FC37-MobileNetV2-0.92.hdf5")
   

   
def lambda_handler(event, context):
    str1 = event['body']
    
    
    ind1 = str1.find("base64") 
    str1 = str1[ind1+6::]
    str1=str1.replace("%2C","")
    str1=str1.replace("%2F","/")
    str1=str1.replace("%2B","+")
    str1=str1.replace("%3D","=")
  
    print(event['body'])
    imgdata = base64.b64decode(str1)
    
    img1=io.BytesIO(imgdata)
    img = Image.open(img1)   
    IMG_W = 224
    IMG_H = 224
    IMG_SHAPE = (IMG_H, IMG_W, 3)
    TARGET_SIZE = (IMG_H, IMG_W)
    image_batch = []
    img =img.resize((224,224),Image.ANTIALIAS)
    strkey='african-american-1436661_1920.jpg'
   # cl.download_file(my_bucket,strkey,'/tmp/african-american-1436661_1920.jpg')
    #img = load_img('/tmp/african-american-1436661_1920.jpg', target_size=TARGET_SIZE)
    x = np.array(img) / 255.0
    x = x.reshape(x.shape)
    image_batch.append(x)
    
    # predict labels: batch_size will handle large amount of images
    preds = model.predict(np.array(image_batch), batch_size=64, verbose=1)
  
    # convert labels to 0, 1 integers.
    preds = np.round(preds,2)
    b=preds.tolist()
    return {
        'statusCode': 200,
        'headers': {
            'Access-Control-Allow-Headers': 'Content-Type',
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': 'OPTIONS,POST,GET'
          #  'Access-Control-Allow-Credentials' : true
        },
        'body': json.dumps(b)
    }
    