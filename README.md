## Overview

Developed two different CNN models modifying Resnet-50 for binary classification to identify whether a patient has COVID-19 or not. 

`Keras` for models and `Tensorflow` for CAM implementation

## Examples

![gradcam_transfer](https://user-images.githubusercontent.com/98493736/208717430-90bb61c0-d80f-4238-a111-4ac44d63ebfe.png)
![gradcam_scratch](https://user-images.githubusercontent.com/98493736/208717468-503005df-a496-409f-b51d-3ae0a6dc44d4.png)


## Procedure

1. Create a CNN by modifying Resnet-18 or Resnet-50 for binary classification (Resnet-50 in this case)

2. Train the CNN from scratch with given training, validation, and testing dataset

3. Train the CNN using transfer learning

4. Compare the two models: Test accuracy of both should be at least 90%+

5. Visualize the two models using various CAM methods - GradCAM and EigenCAM was chosen for this project
