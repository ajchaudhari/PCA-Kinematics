PCA-Kinematics 3D Slicer Module
=============

The 3D Slicer implementation of the method described in  [Foster B, Shaw CB, Boutin RD, Bayne CO, Szabo RM, Joshi AA, and Chaudhari AJ, A Principal Component Analysis-based Framework for Statistical Modeling of Bone Displacement During Wrist Maneuvers, Journal of Biomechanics, https://doi.org/10.1016/j.jbiomech.2019.01.030](https://www.sciencedirect.com/science/article/pii/S0021929019300661). 

An example dataset of the training data is provided in the "Example_Data" folder. 

Please email Brent Foster at bhfoster@ucdavis.edu or Abhijit Chaudhari at ajchaudhari@ucdavis.edu with any questions, comments, or suggestions. 

Purpose 
------- 

Tools for analyzing wrist bone displacement from MRI or CT images. The first 3D Slicer module (Create_Training_Data.py) takes a folder of binary images as an input and outputs the training data for creating the bone displacement model. The second 3D Slicer module (PCA_Kinematics.py) takes a folder of the training data and constructs the bone displacement model.


Installation
------- 
Install the 3D Slicer program if needed. The software is free and open source with downloads provided on https://www.slicer.org/

After installation of 3D Slicer, download the two ".py" files from this GitHub repository (or download the entire repository). Put it in a folder on your computer (an empty folder is recommended for faster loading).

Next, open the 3D Slicer and go to "Edit", then "Application Settings", then "Modules".

Click on the "Add" button on the right to add an additional module path. 

Select the folder where the ".py" files are saved on your local computer.

Restart 3D Slicer.

You should now see a folder named "Wrist PCA-Kinematics" with the modules inside. If not, use the search button or the drop down in the top left in 3D Slicer. 



Input Data 
------- 

<p>
    <img src="Documentation/Input Example Gif.gif" alt>
    <em> Example segmented MR image of the wrist. The module takes a folder of these segmented images (with the same unique label for each bone) as an input. </em>
</p>


Module 1: Create Training Data
------- 

<p>
    <img src="Documentation/Module 1 Example.gif" alt>
    <em> Using a folder of segmented images (which have the same unique label for each bone) as an input, the model creates the training data using the procedure described in the paper. </em>
</p>

Module 2: Create Bone Displacement Model
------- 

<p>
    <img src="Documentation/Module 2 Example.gif" alt>
    <em> The second module takes the output of module 1 (the training data) and creates the bone displacement model using the procedure described in the paper. </em>
</p>

Eigenspace Interpolation
------- 

<p>
    <img src="Documentation/Eigenspace Interpolation.gif" alt>
    <em> Module 2 also has the ability to fit the created bone displacement model to a set of surfaces (which must be in correspondence to the model) and then can interpolate between them in eigenspace. This may allow for more realistic displacement patterns for individual joint modeling. The best fitted model coeffients are also saved to a text file if further analysis is needed. </em>
</p>

