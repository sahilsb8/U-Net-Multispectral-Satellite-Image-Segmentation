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
<img width="700" alt="Google Earth Engine Screenshot" src="https://github.com/sahilsb8/U-Net-Multispectral-Satellite-Image-Segmentation/assets/56041069/ac1b3601-a894-46bf-8a11-0b60bc67cbdf" align="center">
</div>

The images chosen for creation of this specialised dataset were multispectral in nature to potentially harness the infrared capability of satellite imaging. The chosen images had 6 bands and were acquired in the TIFF format which is a vector rasterized format which can hold multiple channel values per pixel.

### Conversion to RGB or IR Images
After acquiring the TIFF files for the selected regions and bands from the Google Earth Engine we can further process them into PNG files with RGB channels or IR channels. This can be done using the Conversion_of_image.ipynb or the Conversion_of_image_nir.ipynb where the former converts the TIFF files into RGB whereas the latter into IR. 

<div align="center">
<img width="400" align="center" alt="RGB Image" src="https://github.com/sahilsb8/U-Net-Multispectral-Satellite-Image-Segmentation/assets/56041069/398ae0dd-0ca1-4674-a241-0c9586e87c23"><img width="400" align="center" alt="Corresponding IR Image" src="https://github.com/sahilsb8/U-Net-Multispectral-Satellite-Image-Segmentation/assets/56041069/4fdb8728-854e-49c1-98bc-58b67891e448">
</div>

### LabelME

There is extensive documentation available for the [LabelMe](https://github.com/labelmeai/labelme) Tool which covers everything from installation to its multiple usecase usage. 

<div align="center">
![SCR-20240526-ukgb](https://github.com/sahilsb8/U-Net-Multispectral-Satellite-Image-Segmentation/assets/56041069/513e8c41-b063-4072-a216-f2b3e43d71a9)
</div>
