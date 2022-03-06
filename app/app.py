import os
import cv2
import datetime
import numpy as np
from pymongo import MongoClient
from keras.preprocessing import image
from keras.models import Sequential,load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Conv2D,MaxPooling2D,Flatten,Dense

client = MongoClient(f"mongodb://{os.environ['MONGO_INITDB_ROOT_USERNAME']}:{os.environ['MONGO_INITDB_ROOT_PASSWORD']}@{os.environ['MONGODB_HOSTNAME']}:27017/entities?authSource=admin&readPreference=primary&appname=aieng&directConnection=true&ssl=false")
db = client.entities

def mongo(data, collection='face_mask_detection'):
    db.face_mask_detection.insert_one(data) if collection == 'face_mask_detection' else db.people_detection.insert_one(data)

def ai_engine():
    model=Sequential()
    model.add(Conv2D(32,(3,3),activation='relu',input_shape=(150,150,3)))
    model.add(MaxPooling2D() )
    model.add(Conv2D(32,(3,3),activation='relu'))
    model.add(MaxPooling2D() )
    model.add(Conv2D(32,(3,3),activation='relu'))
    model.add(MaxPooling2D() )
    model.add(Flatten())
    model.add(Dense(100,activation='relu'))
    model.add(Dense(1,activation='sigmoid'))

    model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
    train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
    test_datagen = ImageDataGenerator(rescale=1./255)
    training_set = train_datagen.flow_from_directory('./data/train', target_size=(150,150), batch_size=16, class_mode='binary')
    test_set = test_datagen.flow_from_directory('./data/test', target_size=(150,150), batch_size=16, class_mode='binary')
    model_saved=model.fit(training_set, epochs=1, validation_data=test_set,)
    model.save('./data/mymodel.h5',model_saved)

    mymodel=load_model('./data/mymodel.h5')
    test_image=image.load_img(r'./data/test/with_mask/1-with-mask.jpg',
                            target_size=(150,150,3))
    test_image
    test_image=image.img_to_array(test_image)
    test_image=np.expand_dims(test_image,axis=0)
    mymodel.predict(test_image)[0][0]

    mymodel=load_model('./data/mymodel.h5')

    cap=cv2.VideoCapture(0)
    face_cascade=cv2.CascadeClassifier('./data/haarcascade_frontalface_default.xml')
    pedestrian_cascade = cv2.CascadeClassifier('./data/haarcascade_fullbody.xml')

    while cap.isOpened():
        ret, frames = cap.read()   
        pedestrians = pedestrian_cascade.detectMultiScale( frames, 1.1, 1)
        # To draw a rectangle in each pedestrians
        for (x,y,w,h) in pedestrians:
            cv2.rectangle(frames,(x,y),(x+w,y+h),(0,255,0),2)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frames, 'Person', (x + 6, y - 6), font, 0.5, (0, 255, 0), 1)
            cv2.imshow('Pedestrian detection', frames)
            mongo({'event': 'PEDESTRIAN', 'timestamp': datetime.datetime.now().timestamp(), 'location': 'Splitaallestrasse 21'},
            collection='pedstrian_detection')
            print(datetime.datetime.now().timestamp(), 'pedestrian')
        # Wait for Enter key to stop
        if cv2.waitKey(33) == 13:
            break   
        _,img=cap.read()
        face=face_cascade.detectMultiScale(img,scaleFactor=1.1,minNeighbors=4)
        for(x,y,w,h) in face:
            face_img = img[y:y+h, x:x+w]
            cv2.imwrite('./data/temp.jpg',face_img)
            test_image=image.load_img('./data/temp.jpg',target_size=(150,150,3))
            test_image=image.img_to_array(test_image)
            test_image=np.expand_dims(test_image,axis=0)
            pred=mymodel.predict(test_image)[0][0]
            if pred==1:
                cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),3)
                cv2.putText(img,'NO MASK',((x+w)//2,y+h+20),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),3)
                print(datetime.datetime.now().timestamp(), 'NO_MASK')
                mongo({'event': 'NO_MASK', 'timestamp': datetime.datetime.now().timestamp(), 'location': 'Splitaallestrasse 21'})
            else:
                cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),3)
                cv2.putText(img,'MASK',((x+w)//2,y+h+20),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),3)
            datet=str(datetime.datetime.now())
            cv2.putText(img,datet,(400,450),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1)
            
        cv2.imshow('img',img)
        
        if cv2.waitKey(1)==ord('q'):
            break
        
    cap.release()
    cv2.destroyAllWindows()

def main():
    ai_engine()

if __name__ == "__main__":
    main()