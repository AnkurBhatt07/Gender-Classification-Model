import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import cv2
import sklearn
import pickle 

haar = cv2.CascadeClassifier("./model/haarcascade_frontalface_default.xml")
pca_dict = pickle.load(open("./model/pca_dict.pickle" , mode = 'rb'))
pca_model = pca_dict['pca']
mean_face = pca_dict['mean_face']
mean_face = mean_face.values
svm_model = pickle.load(open("./model/model_svm.pickle" , mode = 'rb'))


def get_gender_prediction(image):
    global mean_face
    img_rgb = cv2.cvtColor(image , cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(image , cv2.COLOR_BGR2GRAY)

    mean_face = mean_face.reshape((1,10000))

    
    faces = haar.detectMultiScale(gray , scaleFactor=1.5 , minNeighbors=3)
    predictions = []
    for x , y , w , h in faces :
        roi = gray[y:y+h , x : x+w]
        roi = roi / 255.0
        if roi.shape[0] > 100:
            roi_resize = cv2.resize(roi , (100,100), interpolation=cv2.INTER_AREA)
        elif roi.shape[0] < 100:
            roi_resize = cv2.resize(roi , (100,100), interpolation=cv2.INTER_CUBIC)

        else:
            roi_resize = roi 

        flatten = roi_resize.reshape((1,10000))

        # Subtract from the mean face 

        roi_mean = flatten - mean_face 

        # Get Eigen Image 
        roi_mean = pd.DataFrame(roi_mean , columns = [f"pixel_{i}" for i in range(10000)])

        eigen_image = pca_model.transform(roi_mean)

        # Inverse transform 

        eigen_inverse = pca_model.inverse_transform(eigen_image)

        
        # Get the model prediction 

        gender = svm_model.predict(eigen_image)
        prediction_probability = svm_model.predict_proba(eigen_image)
        gender_probability = prediction_probability.max()

        print(f'{gender[0]} : {np.round(gender_probability*100,2)}%')

        text = f"{gender[0]} : {np.round(gender_probability*100 , 2)}%"

        # Defining different colors for the genders
        if gender[0] == 'male':
            color = (255,255,0)
        else:
            color = (255,0,255)

        cv2.rectangle(img_rgb , (x,y) , (x+w , y+h) , color , 2)

        
        cv2.putText(img_rgb , text  , (x,y) , fontFace = cv2.FONT_HERSHEY_PLAIN , fontScale=1 , color = color , thickness = 1 )

        output = {"roi":roi , 
                  "eigen_image" : eigen_image,
                  "eig_inverse":eigen_inverse , 
                  'gender_prediction':gender , 
                  "prediction_prob" : gender_probability
                  }

        predictions.append(output)

    # plt.figure(figsize = (10,10))
    # plt.imshow(img_rgb)
    # plt.axis("off")
    # plt.show()


    return img_rgb,predictions



def face_info(predictions):
    for i in range(len(predictions)):
        face_gray = predictions[i]['roi']
        face_eigen_inv = predictions[i]['eig_inverse']
        face_eigen_inv = face_eigen_inv.reshape((100,100))

        plt.subplot(1,2,1)
        plt.imshow(face_gray , cmap='gray')
        plt.title("Grayscale Image")
        plt.axis('off')
        
        plt.subplot(1,2,2)
        plt.imshow(face_eigen_inv , cmap = 'gray')
        plt.title("Eigen Image")
        plt.axis('off')

        print(f"Predicted Gender = {predictions[i]['gender_prediction']}")
        print(f"Predicted Score = {predictions[i]['prediction_prob'] * 100}")

        print("-"*100)
    
        plt.show()






        