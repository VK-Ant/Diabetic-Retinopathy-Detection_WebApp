import gradio as gr
import cv2
import tensorflow as tf


print("Gradio Version",gr.__version__)


def predict_input_image(img):

    '''Predict input images'''
    
    model = tf.keras.models.load_model(r"/home/vk/Desktop/Retinopathy_Detection_Webapp/End2End_CNNModel_api_webapp/models_30epoch/model00000002-0.9054726362228394.h5") 
    Retina_classes:str = ['DR', 'No_DR']
    img_resize = img.reshape(-1,224,224,3)
    prediction:float=model.predict(img_resize)[0]
    return {Retina_classes[i]: float(prediction[i]) for i in range(2)}

def  GUI():

    '''1. Gradio - Graphical user interface development tool and easy to access anyone

    2. Instead of HTML,CSS,JS -> I am using Gradio'''
   
    #Input shape represent the gradio
    image = gr.inputs.Image(shape=(224,224))
    #Number of classes using Gradio
    label = gr.outputs.Label(num_top_classes=2)
    #All are placing one site! and serve browser
    gr.Interface(fn=predict_input_image, inputs=image, outputs=label,interpretation='default').launch(debug='True')

GUI()