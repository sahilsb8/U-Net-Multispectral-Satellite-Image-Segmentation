# U-Net-Multispectral-Satellite-Image-Segmentation

<div align="center">
<img width="894" alt="SCR-20240525-nkkv" src="https://github.com/sahilsb8/U-Net-Multispectral-Satellite-Image-Segmentation/assets/56041069/a4397099-a0eb-4797-a8fb-479d1a730a39">
</div>

## Description
This repository uses Keras based implementation of U-Net to perform feature detection on satellite images of Indian terrain and buildup focused dataset specially created by us.

## Dataset
- The images were acquired from Google Earth Engine from the Sentinel-2 dataset with a resolution of 5m and contains 6 bands of Red, Green, Blue, NIR, SWIR1, SWIR2.
- Over a 106 images were annotated of Indian landscapes of terrains such as Heavy Urban areas, Farmlands, Coastal areas, Suburban areas, River-bank towns and mountainous regions. These files are located in the dataset/images directory.
- They were annotated using the [LabelMe](https://github.com/labelmeai/labelme) Tool, use of which in our case is explained further.
- Resolution for satellite images s 16-bit. However, mask-files are 8-bit.
- Also available are correctly annotated images of each satellite image location, called mask. These files contain information 4 different classes: Builtup, Vegetation, Water and Undefined. Builtup being defined by the colour red(#800000), Water being defined by the colour yellow(#808000), Vegetation being defined by the colour green(#008000) and Undefined being defined by the colour black(#000000).

## Annotation Procedure
### Google Earth Engine
Google Earth Engine serves as a hub for scientific examination and portrayal of geospatial data, catering to a diverse range of users including academic, non-profit, business, and governmental entities. It houses satellite imagery within a publicly accessible data repository, encompassing historical earth images spanning over four decades. This platform was particularly chosen due to its open source and freeware nature and its technical flexibility. It is easy to use and has an excellent coverage of the globe with great resolution and swathe sizes. It’s almost real time nature and multispectral capability further strengthen the decision to use Google Earth Engine.

<div align="center">
<img width="700" alt="Google Earth Engine Screenshot" src="https://github.com/sahilsb8/U-Net-Multispectral-Satellite-Image-Segmentation/assets/56041069/ac1b3601-a894-46bf-8a11-0b60bc67cbdf">
</div>

The images chosen for creation of this specialised dataset were multispectral in nature to potentially harness the infrared capability of satellite imaging. The chosen images had 6 bands and were acquired in the TIFF format which is a vector rasterized format which can hold multiple channel values per pixel.

### Conversion to RGB or IR Images
After acquiring the TIFF files for the selected regions and bands from the Google Earth Engine we can further process them into PNG files with RGB channels or IR channels. This can be done using the Conversion_of_image.ipynb or the Conversion_of_image_nir.ipynb where the former converts the TIFF files into RGB whereas the latter into IR. 

<div align="center">
<img width="400" alt="RGB Image" src="https://github.com/sahilsb8/U-Net-Multispectral-Satellite-Image-Segmentation/assets/56041069/398ae0dd-0ca1-4674-a241-0c9586e87c23"><img width="400" alt="Corresponding IR Image" src="https://github.com/sahilsb8/U-Net-Multispectral-Satellite-Image-Segmentation/assets/56041069/4fdb8728-854e-49c1-98bc-58b67891e448">
</div>

### LabelME

There is extensive documentation available for the [LabelMe](https://github.com/labelmeai/labelme) Tool which covers everything from installation to its multiple usecase usage. 

<div align="center">
  <img alt="LabelME Screenshot" width="700" src="https://github.com/sahilsb8/U-Net-Multispectral-Satellite-Image-Segmentation/assets/56041069/513e8c41-b063-4072-a216-f2b3e43d71a9">
</div>

After annotating the images save them as JSON files and apply the following functions to convert the JSON files to corresponding the png Mask file. A batch file can be writted to convert multiple files at once according to particular use cases. The following can be used to install LabelME in a python environment titled labelme. 

```python
# python3
conda create --name=labelme python=3
source activate labelme
# conda install -c conda-forge pyside2
# conda install pyqt
# pip install pyqt5  # pyqt5 can be installed via pip on python3
pip install labelme
# or you can install everything by conda command
# conda install labelme -c conda-forge
```

Now to run our environment and the mask conversion script we shall use the following code

```python
conda info --envs
Conda activate labelme
cd directory/with/image
labelme_export_json annotated_image_file_name.json -o mask_file_name
```
## Training the Model

U-Nets were introduced for biomedical image segmentation and have proven to be an effective model for image segmentation in domains other than medicine and in our case for satellite image segmentation. The model used in this project is defined in UNET_RGB.ipynb and UNET_IR.ipynb. They are effectively the same, the only difference being the type of input data being fed to them. One being trained on RGB whereas the other on IR images. It uses transpose convolution layers for upsampling and batch normalization between layers. The summary of the model is given below.

```
__________________________________________________________________________________________________
 Layer (type)                Output Shape                 Param #   Connected to                  
==================================================================================================
 input_1 (InputLayer)        [(None, 512, 512, 3)]        0         []                            
                                                                                                  
 conv2d (Conv2D)             (None, 512, 512, 16)         448       ['input_1[0][0]']             
                                                                                                  
 dropout (Dropout)           (None, 512, 512, 16)         0         ['conv2d[0][0]']              
                                                                                                  
 conv2d_1 (Conv2D)           (None, 512, 512, 16)         2320      ['dropout[0][0]']             
                                                                                                  
 max_pooling2d (MaxPooling2  (None, 256, 256, 16)         0         ['conv2d_1[0][0]']            
 D)                                                                                               
                                                                                                  
 conv2d_2 (Conv2D)           (None, 256, 256, 32)         4640      ['max_pooling2d[0][0]']       
                                                                                                  
 dropout_1 (Dropout)         (None, 256, 256, 32)         0         ['conv2d_2[0][0]']            
                                                                                                  
 conv2d_3 (Conv2D)           (None, 256, 256, 32)         9248      ['dropout_1[0][0]']           
                                                                                                  
 max_pooling2d_1 (MaxPoolin  (None, 128, 128, 32)         0         ['conv2d_3[0][0]']            
 g2D)                                                                                             
                                                                                                  
 conv2d_4 (Conv2D)           (None, 128, 128, 64)         18496     ['max_pooling2d_1[0][0]']     
                                                                                                  
 dropout_2 (Dropout)         (None, 128, 128, 64)         0         ['conv2d_4[0][0]']            
                                                                                                  
 conv2d_5 (Conv2D)           (None, 128, 128, 64)         36928     ['dropout_2[0][0]']           
                                                                                                  
 max_pooling2d_2 (MaxPoolin  (None, 64, 64, 64)           0         ['conv2d_5[0][0]']            
 g2D)                                                                                             
                                                                                                  
 conv2d_6 (Conv2D)           (None, 64, 64, 128)          73856     ['max_pooling2d_2[0][0]']     
                                                                                                  
 dropout_3 (Dropout)         (None, 64, 64, 128)          0         ['conv2d_6[0][0]']            
                                                                                                  
 conv2d_7 (Conv2D)           (None, 64, 64, 128)          147584    ['dropout_3[0][0]']           
                                                                                                  
 max_pooling2d_3 (MaxPoolin  (None, 32, 32, 128)          0         ['conv2d_7[0][0]']            
 g2D)                                                                                             
                                                                                                  
 conv2d_8 (Conv2D)           (None, 32, 32, 256)          295168    ['max_pooling2d_3[0][0]']     
                                                                                                  
 dropout_4 (Dropout)         (None, 32, 32, 256)          0         ['conv2d_8[0][0]']            
                                                                                                  
 conv2d_9 (Conv2D)           (None, 32, 32, 256)          590080    ['dropout_4[0][0]']           
                                                                                                  
 max_pooling2d_4 (MaxPoolin  (None, 16, 16, 256)          0         ['conv2d_9[0][0]']            
 g2D)                                                                                             
                                                                                                  
 conv2d_10 (Conv2D)          (None, 16, 16, 512)          1180160   ['max_pooling2d_4[0][0]']     
                                                                                                  
 dropout_5 (Dropout)         (None, 16, 16, 512)          0         ['conv2d_10[0][0]']           
                                                                                                  
 conv2d_11 (Conv2D)          (None, 16, 16, 512)          2359808   ['dropout_5[0][0]']           
                                                                                                  
 conv2d_transpose (Conv2DTr  (None, 32, 32, 256)          524544    ['conv2d_11[0][0]']           
 anspose)                                                                                         
                                                                                                  
 concatenate (Concatenate)   (None, 32, 32, 512)          0         ['conv2d_transpose[0][0]',    
                                                                     'conv2d_9[0][0]']            
                                                                                                  
 conv2d_12 (Conv2D)          (None, 32, 32, 256)          1179904   ['concatenate[0][0]']         
                                                                                                  
 dropout_6 (Dropout)         (None, 32, 32, 256)          0         ['conv2d_12[0][0]']           
                                                                                                  
 conv2d_13 (Conv2D)          (None, 32, 32, 256)          590080    ['dropout_6[0][0]']           
                                                                                                  
 conv2d_transpose_1 (Conv2D  (None, 64, 64, 128)          131200    ['conv2d_13[0][0]']           
 Transpose)                                                                                       
                                                                                                  
 concatenate_1 (Concatenate  (None, 64, 64, 256)          0         ['conv2d_transpose_1[0][0]',  
 )                                                                   'conv2d_7[0][0]']            
                                                                                                  
 conv2d_14 (Conv2D)          (None, 64, 64, 128)          295040    ['concatenate_1[0][0]']       
                                                                                                  
 dropout_7 (Dropout)         (None, 64, 64, 128)          0         ['conv2d_14[0][0]']           
                                                                                                  
 conv2d_15 (Conv2D)          (None, 64, 64, 128)          147584    ['dropout_7[0][0]']           
                                                                                                  
 conv2d_transpose_2 (Conv2D  (None, 128, 128, 64)         32832     ['conv2d_15[0][0]']           
 Transpose)                                                                                       
                                                                                                  
 concatenate_2 (Concatenate  (None, 128, 128, 128)        0         ['conv2d_transpose_2[0][0]',  
 )                                                                   'conv2d_5[0][0]']            
                                                                                                  
 conv2d_16 (Conv2D)          (None, 128, 128, 64)         73792     ['concatenate_2[0][0]']       
                                                                                                  
 dropout_8 (Dropout)         (None, 128, 128, 64)         0         ['conv2d_16[0][0]']           
                                                                                                  
 conv2d_17 (Conv2D)          (None, 128, 128, 64)         36928     ['dropout_8[0][0]']           
                                                                                                  
 conv2d_transpose_3 (Conv2D  (None, 256, 256, 32)         8224      ['conv2d_17[0][0]']           
 Transpose)                                                                                       
                                                                                                  
 concatenate_3 (Concatenate  (None, 256, 256, 64)         0         ['conv2d_transpose_3[0][0]',  
 )                                                                   'conv2d_3[0][0]']            
                                                                                                  
 conv2d_18 (Conv2D)          (None, 256, 256, 32)         18464     ['concatenate_3[0][0]']       
                                                                                                  
 dropout_9 (Dropout)         (None, 256, 256, 32)         0         ['conv2d_18[0][0]']           
                                                                                                  
 conv2d_19 (Conv2D)          (None, 256, 256, 32)         9248      ['dropout_9[0][0]']           
                                                                                                  
 conv2d_transpose_4 (Conv2D  (None, 512, 512, 16)         2064      ['conv2d_19[0][0]']           
 Transpose)                                                                                       
                                                                                                  
 concatenate_4 (Concatenate  (None, 512, 512, 32)         0         ['conv2d_transpose_4[0][0]',  
 )                                                                   'conv2d_1[0][0]']            
                                                                                                  
 conv2d_20 (Conv2D)          (None, 512, 512, 16)         4624      ['concatenate_4[0][0]']       
                                                                                                  
 dropout_10 (Dropout)        (None, 512, 512, 16)         0         ['conv2d_20[0][0]']           
                                                                                                  
 conv2d_21 (Conv2D)          (None, 512, 512, 16)         2320      ['dropout_10[0][0]']          
                                                                                                  
 conv2d_22 (Conv2D)          (None, 512, 512, 4)          68        ['conv2d_21[0][0]']           
                                                                                                  
==================================================================================================
Total params: 7775652 (29.66 MB)
Trainable params: 7775652 (29.66 MB)
Non-trainable params: 0 (0.00 Byte)
__________________________________________________________________________________________________
```
Note: This model was trained for a 100 epochs and a batch size of 16 on a V100 GPU. The train test split was 80:20.

# Class Statistics Calculation

The statistics and imbalances of the dataset can be found out using the Class_stats.ipynb and AverageResCalulator.ipynb. The Class_stats uses pixel wise area calculation to find the class label imbalances. In the provided dataset the following statistics were found.

<div align="center">
<img alt="Class Stats" width="700" src="https://github.com/sahilsb8/U-Net-Multispectral-Satellite-Image-Segmentation/assets/56041069/3e9bdd31-7557-40b4-a556-9d73c283efee">
</div>

The other notebook titled AverageResCalulator.ipynb uses siple averaging technique to calculate the average pixel height and width of our dataset. In our dataset it came out to be, width = 1608 px and height = 940 px. These statistics are important to maintain the balance of the dataset and provide in the conducted research. 

### TileMaker

There is also a Python Notebook provided which can be used to create tiles or patches of desired size and save them natively in PNG formats. These patches although ultimately not useful for our model it can be used to save memory by not patchifying in the runtime. 
