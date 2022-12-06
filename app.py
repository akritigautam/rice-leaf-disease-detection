import streamlit as st
import tensorflow as tf
st.set_option('deprecation.showfileUploaderEncoding',False)
def load_model():
    model=tf.keras.models.load_model("mymodel2.hdf5")
    return model
model=load_model()
st.write("Leaf detection")
file=st.file_uploader("Please upload image",type=["jpg","png","jpeg"])
import cv2
from PIL import Image,ImageOps
import numpy as np
def import_and_predict(image_data,model):
    size=(224,224)
    image=ImageOps.fit(image_data,size,Image.ANTIALIAS)
    image=np.asarray(image)
    image_reshape=image[np.newaxis,...]
    prediction=model.predict(image_reshape)
if file is None:
    st.text("Please upload an image file")
else:
    image=Image.open(file)
    st.image(image,use_column_width=True)
    predictions=import_and_predict(image,model)
    class_names=['Brown Spot','Leaf smut','Bacterial leaf blight']
    strings="The disease detected in this leaf is : "+class_names[np.argmax(predictions)]
    st.success(strings)    

