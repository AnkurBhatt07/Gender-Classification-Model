import os 
import cv2 
from flask import render_template , request 
from app.FRM_pipeline import get_gender_prediction , face_info
import matplotlib.image as matimg 


UPLOAD_FOLDER = 'static/upload'


def index():
    return render_template('index.html')


def app():
    return render_template('app.html')

def genderapp():
    if request.method == "POST":
        f = request.files['image_name']
        filename = f.filename

        # save the image in the upload folder

        path = os.path.join(UPLOAD_FOLDER,filename)

        f.save(path)

        # Get predictions 

        image = cv2.imread(path)
        pred_image , predictions = get_gender_prediction(image)
        pred_filename = 'prediction_image.jpg'

        cv2.imwrite(f'./static/predict/{pred_filename}' , cv2.cvtColor(pred_image, cv2.COLOR_RGB2BGR))


        # generate report 

        report = []
        for i , obj in enumerate(predictions):
            gray_image = obj['roi']
            eigen_image = obj['eig_inverse'].reshape(100,100)
            gender_name = obj['gender_prediction']
            score = round(obj['prediction_prob']*100,2)

            # save grayscale and eigne in predict folder
            gray_image_name = f'roi_{i}.jpg'
            eig_image_name = f'eigen_{i}.jpg'
            matimg.imsave(f'./static/predict/{gray_image_name}',gray_image,cmap='gray')
            matimg.imsave(f'./static/predict/{eig_image_name}',eigen_image,cmap='gray')
            
            # save report 
            report.append([gray_image_name,
                           eig_image_name,
                           gender_name,
                           score])

        return render_template('gender.html' , fileupload = True , report = report)
    
    return render_template('gender.html' , fileupload = False)

