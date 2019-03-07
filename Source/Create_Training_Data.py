
from __main__ import vtk, qt, ctk, slicer
import EditorLib

import SimpleITK as sitk
import sitkUtils
import numpy as np
import multiprocessing
import timeit

import os


#
# Create_Training_Data
#
class Create_Training_Data:
    def __init__(self, parent):
        import string
        parent.title = "Create PCA Kinematics Training Data"
        parent.categories = ["PCA-Kinematics"]
        parent.contributors = ["Brent Foster (University of California Davis)"]
        parent.helpText = string.Template("""
        
        PURPOSE: Use this 3D Slicer module to create the training data for contructing a bone displacement 
        model using a principal component analysis (PCA) based approach as described in Foster et al Journal of Biomechanics (https://doi.org/10.1016/j.jbiomech.2019.01.030). 
        <br>
        <br>
        INPUT: A folder of segmented images (.nii or .img/.hdr files) with  each bone having the same unique and non-zero label on all the images. 
        <br>
        <br>
        OUTPUT: A folder of PLY surface files which serve as training data for the PCA bone displacement method. (Use the PCA_Kinematics 3D Slicer module with this training data).
        <br>
        <br>
        PROCESSING STEPS: The module has the following steps
        <br>
        (1) Extract the bones from the images and convert the bones to 3D surfaces 
        <br>
        (2) Apply surface smoothing and surface decimation (to reduce the number of surface vertices) 
        <br>
        (3) Using the iterative closest point (ICP) registration algorithm, align the 'reference' bone among all the volunteers
        <br>
        (4) Using the bone surfaces from the first volunteer as reference shapes, align these to the corresponding bone on the other volunteers using ICP registration 
        <br>
        (5) Export the set of surfaces for each volunteer as a PLY file. The folder of exported surfaces is the training data.
        

        """
            ).substitute({
        'a': parent.slicerWikiUrl, 'b': slicer.app.majorVersion, 'c': slicer.app.minorVersion})
        parent.acknowledgementText = """
        Supported by funding from the National Science Foundation (GRFP Grant No. 1650042) \
        and National Institutes of Health (K12 HD051958 and R03 EB015099). 
        
        <br>
        <br>

        Module implemented by Brent Foster. Last updated on February 8, 2019.

        <br>
        <br>
        <br>

        Foster B, Shaw CB, Boutin RD, Bayne CO, Szabo RM, Joshi AA, and Chaudhari AJ, \
        A Principal Component Analysis-based Framework for Statistical Modeling of Bone \
        Displacement During Wrist Maneuvers, Journal of Biomechanics, https://doi.org/10.1016/j.jbiomech.2019.01.030

        """
        parent.index = 0
        self.parent = parent

class Create_Training_DataWidget:
    def __init__(self, parent=None):
        self.parent = parent
        self.logic = None
        self.ImageNode = None
        self.icp_mode = 'Rigid' # Default for the ICP registration mode
        self.directory_path = []
        self.output_directory_path = []

        # Font settings for the module
        self.font_type = "Arial"
        self.font_size = 12

    def setup(self):
        frame = qt.QFrame()
        frameLayout = qt.QFormLayout()
        frame.setLayout(frameLayout)
        self.parent.layout().addWidget(frame)

        # Choose directory button to choose the folder for saving the registered image 
        self.directoryButton = qt.QPushButton("Choose folder with the segmented images")
        self.directoryButton.setFont(qt.QFont(self.font_type, self.font_size))
        self.directoryButton.toolTip = "Choose a folder of .nii or .img/.hdr images of the segmented bones."
        frameLayout.addWidget(self.directoryButton)
        self.directoryButton.connect('clicked()', self.onDirectoryButtonClick)
        frameLayout.addRow(self.directoryButton)

        # Choose the Output folder button to choose the folder for saving the registered image 
        self.outputDirectoryButton = qt.QPushButton("Choose folder to save the output training data to")
        self.outputDirectoryButton.setFont(qt.QFont(self.font_type, self.font_size))
        self.outputDirectoryButton.toolTip = "Choose a folder to save the output"
        frameLayout.addWidget(self.outputDirectoryButton)
        self.outputDirectoryButton.connect('clicked()', self.onOutputDirectoryButtonClick)
        frameLayout.addRow(self.outputDirectoryButton)

        # ICP Registration Collapse button
        self.ICPCollapsibleButton = ctk.ctkCollapsibleButton()
        self.ICPCollapsibleButton.setFont(qt.QFont(self.font_type, self.font_size))
        self.ICPCollapsibleButton.text = "Iterative Closest Point Registration"
        self.ICPCollapsibleButton.collapsed = True # Default is to not show     
        frameLayout.addWidget(self.ICPCollapsibleButton) 

        # Layout within the ICP collapsible button
        self.ICPFormLayout = qt.QFormLayout(self.ICPCollapsibleButton)

        # Slider for Maximum Iteration Number for ICP registration    
        self.label = qt.QLabel()
        self.label.setFont(qt.QFont(self.font_type, self.font_size))
        self.label.setText("Maximum Iteration Number: ")
        self.label.setToolTip("Select the maximum iteration number for the ICP registration.")
        self.IterationSlider = ctk.ctkSliderWidget()
        self.IterationSlider.setToolTip("Select the maximum iteration number for the ICP registration.")
        self.IterationSlider.minimum = 1
        self.IterationSlider.maximum = 200
        self.IterationSlider.value = 100
        self.IterationSlider.singleStep = 5
        self.IterationSlider.tickInterval = 1
        self.IterationSlider.decimals = 0
        self.IterationSlider.connect('valueChanged(double)', self.onIterationSliderChange)
        self.ICPFormLayout.addRow(self.label, self.IterationSlider)        
        self.IterationNumber = self.IterationSlider.value # Set default value

        # Slider for Number of Landmarks for ICP      
        self.label = qt.QLabel()
        self.label.setFont(qt.QFont(self.font_type, self.font_size))
        self.label.setText("ICP Landmark Number: ")
        self.label.setToolTip("Select the number of landmarks per surface for the ICP registration.")
        self.LandmarkSlider = ctk.ctkSliderWidget()
        self.LandmarkSlider.setToolTip("Select the number of landmarks per surface for the ICP registration.")
        self.LandmarkSlider.minimum = 50
        self.LandmarkSlider.maximum = 1000
        self.LandmarkSlider.value = 500
        self.LandmarkSlider.singleStep = 10
        self.LandmarkSlider.tickInterval = 1
        self.LandmarkSlider.decimals = 0
        self.LandmarkSlider.connect('valueChanged(double)', self.onLandmarkSliderChange)
        self.ICPFormLayout.addRow(self.label, self.LandmarkSlider)        
        self.LandmarkNumber = self.LandmarkSlider.value # Set default value

        # Slider for Maximum RMS Error for ICP       
        self.label = qt.QLabel()
        self.label.setFont(qt.QFont(self.font_type, self.font_size))
        self.label.setText("ICP Maximum RMS Error: ")
        self.label.setToolTip("Select the maximum root mean square (RMS) error for determining the ICP registration convergence.")
        self.RMS_Slider = ctk.ctkSliderWidget()
        self.RMS_Slider.setToolTip("Select the maximum root mean square (RMS) error for determining the ICP registration convergence.")
        self.RMS_Slider.minimum = 0.0001
        self.RMS_Slider.maximum = 0.05
        self.RMS_Slider.value = 0.01
        self.RMS_Slider.singleStep = 0.01
        self.RMS_Slider.tickInterval = 0.001
        self.RMS_Slider.decimals = 3
        self.RMS_Slider.connect('valueChanged(double)', self.onRMS_SliderChange)
        self.ICPFormLayout.addRow(self.label, self.RMS_Slider)
        self.RMS_Number = self.RMS_Slider.value # Set default value
        
        # Slider for choosing the reference bone      
        self.label = qt.QLabel()
        self.label.setFont(qt.QFont(self.font_type, self.font_size))
        self.label.setText("Reference Bone Label: ")
        self.label.setToolTip("Select the label of the bone to keep static. This bone will be registered among all the volunteers. Note: Please choose the folder with the segmented images first.")
        self.Ref_Bone_Slider = ctk.ctkSliderWidget()
        self.Ref_Bone_Slider.setToolTip("Select the label of the bone to keep static. This bone will be registered among all the volunteers. Note: Please choose the folder with the segmented images first.")    
        self.Ref_Bone_Slider.minimum = 0
        self.Ref_Bone_Slider.maximum = 0 
        self.Ref_Bone_Slider.value = 0 
        self.Ref_Bone_Slider.singleStep = 1
        self.Ref_Bone_Slider.tickInterval = 1
        self.Ref_Bone_Slider.decimals = 0
        self.Ref_Bone_Slider.connect('valueChanged(double)', self.onRef_Bone_SliderChange)
        self.ICPFormLayout.addRow(self.label, self.Ref_Bone_Slider)        
        self.ref_label = int(self.Ref_Bone_Slider.value) # Set default value

        # Radial buttons for the ICP parameters
        self.radial_button_1 = qt.QRadioButton("Similarity")
        self.radial_button_1.setFont(qt.QFont(self.font_type, self.font_size))
        self.radial_button_1.toggled.connect(self.onICPModeSelect_1)
        self.ICPFormLayout.addWidget(self.radial_button_1)

        self.radial_button_2 = qt.QRadioButton("Rigid")
        self.radial_button_2.setFont(qt.QFont(self.font_type, self.font_size))
        self.radial_button_2.setChecked(True)
        self.radial_button_2.toggled.connect(self.onICPModeSelect_2)
        self.ICPFormLayout.addWidget(self.radial_button_2)

        self.radial_button_3 = qt.QRadioButton("Affine")
        self.radial_button_3.toggled.connect(self.onICPModeSelect_3)
        self.ICPFormLayout.addWidget(self.radial_button_3)

        # Text input for choosing which image labels to use      
        self.label = qt.QLabel()
        self.label.setFont(qt.QFont(self.font_type, self.font_size))
        self.label.setText("Image Labels: ")
        self.label.setToolTip("Choose several labels to use for creating the training data. Value of -1 uses the default 1 through 9. Otherwise, input should be similar to '1,2,3' to use labels one, two, and three. ")
        self.lineedit = qt.QLineEdit()
        self.lineedit.setFont(qt.QFont(self.font_type, self.font_size))
        self.lineedit.setToolTip("Choose several labels to use for creating the training data. Value of -1 uses the default 1 through 9. Otherwise, input should be similar to '1,2,3' to use labels one, two, and three. ")
        self.ICPFormLayout.addRow(self.label, self.lineedit)
        self.lineedit.setText("-1")

        # Bone Smoothing Parameters Collapse Button
        self.SmoothCollapsibleButton = ctk.ctkCollapsibleButton()
        self.SmoothCollapsibleButton.setFont(qt.QFont(self.font_type, self.font_size))
        self.SmoothCollapsibleButton.text = "Smoothing Options"
        self.SmoothCollapsibleButton.collapsed = True # Default is to not show  
        frameLayout.addWidget(self.SmoothCollapsibleButton) 

        # Layout within the smoothing options collapsible button
        self.SmoothFormLayout = qt.QFormLayout(self.SmoothCollapsibleButton)

        # Slider for choosing the number of iterations for bone smoothing     
        self.label = qt.QLabel()
        self.label.setFont(qt.QFont(self.font_type, self.font_size))
        self.label.setText("Smoothing Iterations: ")
        self.label.setToolTip("Select the number of iterations for smoothing the bone surface. Higher iterations will smooth more. Lower iterations will have less smoothing.")
        self.Bone_Smoothing_Its_Slider = ctk.ctkSliderWidget()
        self.Bone_Smoothing_Its_Slider.setToolTip("Select the number of iterations for smoothing the bone surface. Higher iterations will smooth more. Lower iterations will have less smoothing.")
        self.Bone_Smoothing_Its_Slider.minimum = 0
        self.Bone_Smoothing_Its_Slider.maximum = 30
        self.Bone_Smoothing_Its_Slider.value = 10 
        self.Bone_Smoothing_Its_Slider.singleStep = 1
        self.Bone_Smoothing_Its_Slider.tickInterval = 1
        self.Bone_Smoothing_Its_Slider.decimals = 0
        self.Bone_Smoothing_Its_Slider.connect('valueChanged(double)', self.onBone_Smoothing_Its_SliderChange)
        self.SmoothFormLayout.addRow(self.label, self.Bone_Smoothing_Its_Slider)
        self.smoothing_iterations = self.Bone_Smoothing_Its_Slider.value # Set default value
        
        # Slider for choosing the smoothing relaxation factor     
        self.label = qt.QLabel()
        self.label.setFont(qt.QFont(self.font_type, self.font_size))
        self.label.setText("Relaxation Factor: ")
        self.label.setToolTip("Select the relaxation factor for smoothing the bone surfaces. Higher relaxation will smooth the surface more while a lower factor will have less smoothing.")
        self.Bone_Smoothing_Relaxation_Slider = ctk.ctkSliderWidget()
        self.Bone_Smoothing_Relaxation_Slider.setToolTip("Select the relaxation factor for smoothing the bone surfaces. Higher relaxation will smooth the surface more while a lower factor will have less smoothing.")
        self.Bone_Smoothing_Relaxation_Slider.minimum = 0
        self.Bone_Smoothing_Relaxation_Slider.maximum = 1
        self.Bone_Smoothing_Relaxation_Slider.value = 0.4 
        self.Bone_Smoothing_Relaxation_Slider.singleStep = 0.1
        self.Bone_Smoothing_Relaxation_Slider.tickInterval = 0.1
        self.Bone_Smoothing_Relaxation_Slider.decimals = 2
        self.Bone_Smoothing_Relaxation_Slider.connect('valueChanged(double)', self.onBone_Smoothing_Relaxation_SliderChange)
        self.SmoothFormLayout.addRow(self.label, self.Bone_Smoothing_Relaxation_Slider)
        self.relaxation_factor = self.Bone_Smoothing_Relaxation_Slider.value # Set default value

        # Slider for choosing the percentage of bone surface decimation    
        self.label = qt.QLabel()
        self.label.setFont(qt.QFont(self.font_type, self.font_size))
        self.label.setText("Surface Decimation: ")
        self.label.setToolTip("Select the ratio of verticies to remove from the bone surface. This results in a smoother surface and less points for faster computation. Too much could cause surface artifacts. For example, 0.1 removes 10 percent while 0.9 removes 90 percent of points.")
        self.Bone_Decimate_Slider = ctk.ctkSliderWidget()
        self.Bone_Decimate_Slider.setToolTip("Select the ratio of verticies to remove from the bone surface. This results in a smoother surface and less points for faster computation. Too much could cause surface artifacts. For example, 0.1 removes 10 percent while 0.9 removes 90 percent of points.")     
        self.Bone_Decimate_Slider.minimum = 0
        self.Bone_Decimate_Slider.maximum = 0.9
        self.Bone_Decimate_Slider.value = 0
        self.Bone_Decimate_Slider.singleStep = 0.05
        self.Bone_Decimate_Slider.tickInterval = 0.05
        self.Bone_Decimate_Slider.decimals = 2
        self.Bone_Decimate_Slider.connect('valueChanged(double)', self.onBone_Decimate_SliderChange)
        self.SmoothFormLayout.addRow(self.label, self.Bone_Decimate_Slider)        
        self.decimate_surface = self.Bone_Decimate_Slider.value # Set default value    

        # Debugging Collapse button
        self.RenderingCollapsibleButton = ctk.ctkCollapsibleButton()
        self.RenderingCollapsibleButton.setFont(qt.QFont(self.font_type, self.font_size))
        self.RenderingCollapsibleButton.text = "Visualization"
        self.RenderingCollapsibleButton.collapsed = True # Default is to not show   
        frameLayout.addWidget(self.RenderingCollapsibleButton) 

        # Layout within the debug collapsible button
        self.CollapseFormLayout = qt.QFormLayout(self.RenderingCollapsibleButton)

        # Show the registered shapes toggle button
        self.show_registered_shapes = qt.QCheckBox("Show Registered Shapes")
        self.show_registered_shapes.setFont(qt.QFont(self.font_type, self.font_size))
        self.show_registered_shapes.toolTip = "When checked, show each registered bone. Useful for debugging any ICP registration based errors."
        self.show_registered_shapes.checked = False
        self.CollapseFormLayout.addWidget(self.show_registered_shapes) 

        # Show the registered shapes toggle button
        self.show_extracted_shapes = qt.QCheckBox("Show Extracted Shapes")
        self.show_extracted_shapes.setFont(qt.QFont(self.font_type, self.font_size))
        self.show_extracted_shapes.toolTip = "When checked, show the initial surfaces from the images. Useful for debugging errors based on deriving surfaces from the MR images."
        self.show_extracted_shapes.checked = False
        self.CollapseFormLayout.addWidget(self.show_extracted_shapes) 

        # Show the loaded images toggle button
        self.debug_show_images = qt.QCheckBox("Show Loaded Images")
        self.debug_show_images.setFont(qt.QFont(self.font_type, self.font_size))
        self.debug_show_images.toolTip = "When checked, show the loaded images. Useful for debugging if the images are flipped or loading incorrectly."
        self.debug_show_images.checked = False
        self.CollapseFormLayout.addWidget(self.debug_show_images) 


        # Debugging Collapse button
        self.DebugCollapsibleButton = ctk.ctkCollapsibleButton()
        self.DebugCollapsibleButton.setFont(qt.QFont(self.font_type, self.font_size))
        self.DebugCollapsibleButton.text = "Debugging"
        self.DebugCollapsibleButton.collapsed = True # Default is to not show   
        frameLayout.addWidget(self.DebugCollapsibleButton) 

        # Layout within the debug collapsible button
        self.CollapseFormLayout = qt.QFormLayout(self.DebugCollapsibleButton)



        # Show the flip image vertically toggle button
        self.flip_image_vertically = qt.QCheckBox("Flip Vertically")
        self.flip_image_vertically.setFont(qt.QFont(self.font_type, self.font_size))
        self.flip_image_vertically.toolTip = "When checked, flip the images vertically after loading them."
        self.flip_image_vertically.checked = False
        self.CollapseFormLayout.addWidget(self.flip_image_vertically) 

        # Show the flip image horizontally toggle button
        self.flip_image_horizontally = qt.QCheckBox("Flip Horizontally")
        self.flip_image_horizontally.setFont(qt.QFont(self.font_type, self.font_size))
        self.flip_image_horizontally.toolTip = "When checked, flip the images horizontally after loading them."
        self.flip_image_horizontally.checked = False
        self.CollapseFormLayout.addWidget(self.flip_image_horizontally) 

        # Show the save bones separately toggle button
        self.Save_Extracted_Bones_Separately = qt.QCheckBox("Save Extracted Bones Separately")
        self.Save_Extracted_Bones_Separately.setFont(qt.QFont(self.font_type, self.font_size))
        self.Save_Extracted_Bones_Separately.toolTip = "When checked, save each bone surface separately after smoothing and before any registration."
        self.Save_Extracted_Bones_Separately.checked = False
        self.CollapseFormLayout.addWidget(self.Save_Extracted_Bones_Separately) 

        # Show the save bones separately toggle button
        self.Save_Registered_Bones_Separately = qt.QCheckBox("Save Registered Bones Separately")
        self.Save_Registered_Bones_Separately.setFont(qt.QFont(self.font_type, self.font_size))
        self.Save_Registered_Bones_Separately.toolTip = "When checked, save each bone surface separately after smoothing and registration (instead of combining all the bones for each volunteer/position together)."
        self.Save_Registered_Bones_Separately.checked = False
        self.CollapseFormLayout.addWidget(self.Save_Registered_Bones_Separately) 

        # Skip the registration
        self.Skip_Registration = qt.QCheckBox("Skip Registration Steps")
        self.Skip_Registration.setFont(qt.QFont(self.font_type, self.font_size))
        self.Skip_Registration.toolTip = "When checked, extract each bone surface from the image and smooth the surface (this is useful if the user only wishes to extract the bones from the images and smooth them.) Consider using along with the save extracted bones setting."
        self.Skip_Registration.checked = False
        self.CollapseFormLayout.addWidget(self.Skip_Registration) 

        # Don't save the reference bone toggle button
        self.Remove_Ref_Bone = qt.QCheckBox("Remove the reference bone")
        self.Remove_Ref_Bone.setFont(qt.QFont(self.font_type, self.font_size))
        self.Remove_Ref_Bone.toolTip = "When checked, don't save the refernce bone when saving the PLY file. For example, this is useful if using the radius as a reference bone, but if you don't want it to appear in the final model."
        self.Remove_Ref_Bone.checked = False
        self.CollapseFormLayout.addWidget(self.Remove_Ref_Bone) 

        # Choose the number of files to load from the given folder     
        self.label = qt.QLabel()
        self.label.setFont(qt.QFont(self.font_type, self.font_size))
        self.label.setText("Number of images: ")
        self.label.setToolTip("Select how many images to use from the given folder. (Select the folder first). This is useful if want want to use just the first few images to make sure everything runs correctly.")
        self.NumFileSlider = ctk.ctkSliderWidget()
        self.NumFileSlider.setToolTip("Select how many images to use from the given folder. (Select the folder first). This is useful if want want to use just the first few images to make sure everything runs correctly.")
        self.NumFileSlider.minimum = -1
        self.NumFileSlider.maximum = 1
        self.NumFileSlider.value = -1
        self.NumFileSlider.singleStep = 1
        self.NumFileSlider.tickInterval = 1
        self.NumFileSlider.decimals = 0
        self.NumFileSlider.connect('valueChanged(double)', self.onNumSliderChange)
        self.CollapseFormLayout.addRow(self.label, self.NumFileSlider)        
        self.num_files = self.NumFileSlider.value # Set default value

        # Compute button
        self.computeButton = qt.QPushButton("Create The Training Data")
        self.computeButton.setFont(qt.QFont(self.font_type, self.font_size))
        self.computeButton.toolTip = "Run the module and create the training data for the PCA bone displacement model."
        frameLayout.addWidget(self.computeButton)
        self.computeButton.connect('clicked()', self.onCompute)
        self.computeButton.enabled = False

        # Progress Bar (so the user knows how much longer it will take)
        self.progressBar = qt.QProgressBar()
        self.progressBar.setFont(qt.QFont(self.font_type, self.font_size))
        self.progressBar.setValue(0)
        frameLayout.addWidget(self.progressBar)
        self.progressBar.hide()
        
    def onOutputDirectoryButtonClick(self):
        # After clicking the button, let the user choose a directory for saving
        self.output_directory_path = qt.QFileDialog.getExistingDirectory()

        # Update the QT label with the directory path so the user can see it
        try:
            self.outputDirectoryButton.setText('Output:  ' + self.output_directory_path[-35:]) # Only show the last 35 characters for better display in the Slicer GUI
        except:
            self.outputDirectoryButton.setText('Output:  ' + self.output_directory_path) # Show the entire path if it is less than 35 characters long

        # Update the compute button state (to determine if it should be enabled or not)
        self.UpdatecomputeButtonState()

    def onBone_Decimate_SliderChange(self, newValue):
        # Percentage to decimate the bone surface by
        # This results in a smoother surface and less points for faster computation
        # Although too much could cause surface artifacts

        self.decimate_surface = newValue
        
    def onBone_Smoothing_Its_SliderChange(self, newValue):
        # Number of iterations for bone surface smoothing
        self.smoothing_iterations = newValue

    def onBone_Smoothing_Relaxation_SliderChange(self, newValue):
        # Relaxation factor for bone surface smoothing
        self.relaxation_factor = newValue

    def onRef_Bone_SliderChange(self, newValue):
        # The label of the segmented image to consider as a reference bone
        # This bone is held statically between all volunteers by ICP registration
        self.ref_label = int(newValue)

    def onICPModeSelect_1(self,newValue):
        # Set the mode for ICP registration
        if newValue == True:
            self.icp_mode = 'Similarity' 
            print(self.icp_mode)

    def onICPModeSelect_2(self,newValue):
        # Set the mode for ICP registration
        if newValue == True:
            self.icp_mode = 'Rigid' 
            print(self.icp_mode)

    def onICPModeSelect_3(self,newValue):
        # Set the mode for ICP registration
        if newValue == True:
            self.icp_mode = 'Affine' 
            print(self.icp_mode)

    def onRMS_SliderChange(self, newValue):
        self.RMS_Number = newValue

    def onLandmarkSliderChange(self, newValue):
        self.LandmarkNumber = newValue        

    def onIterationSliderChange(self, newValue):
        self.IterationNumber = newValue

    def onNumSliderChange(self, value):
        # Slider for selecting the number of images to use
        # For debugging or if the model generation is too slow
        self.num_files = value

    def onDirectoryButtonClick(self):
        # After clicking the button, let the user choose a directory for saving
        self.directory_path = qt.QFileDialog.getExistingDirectory()

        # Update the QT label with the directory path so the user can see it
        try:
            self.directoryButton.setText('Segmentations: ' + self.directory_path[-35:]) # Only show the last 35 characters for better display in the Slicer GUI
        except:
            self.directoryButton.setText('Segmentations: ' + self.directory_path) # Show the entire path if it is less than 35 characters long

        # Update the slider for selecting the number of files to use (for debugging)
        self.NumFileSlider.maximum = len(os.listdir(self.directory_path)) 

        # Update the "Reference Bone Label" slider (i.e. self.Ref_Bone_Slider) to have a maximum value 
        # Based on the maximum value of the first image loaded

        # Find all the files in the input folder
        self.files = os.listdir(self.directory_path)

        # Use the first file in the folder
        imgReader = self.load_image(str(self.directory_path + '\\' + self.files[0]))

        from vtk.util.numpy_support import vtk_to_numpy
        array = vtk_to_numpy(imgReader.GetOutput().GetPointData().GetScalars())
        self.Ref_Bone_Slider.minimum = np.amin(array) + 1 # Assume the lowest value is the background label
        self.Ref_Bone_Slider.maximum  = np.amax(array)
        self.Ref_Bone_Slider.value = np.amax(array) - 1 # Default should be 9 (the radius bone)

        # Update the compute button state (to determine if it should be enabled or not)
        self.UpdatecomputeButtonState()


    def UpdatecomputeButtonState(self):
        # Enable the 'Compute' button only if there is a selection to the input volume and markup list
        # if self.FixedModelSelector.currentNode() and self.inputSelector.currentNode() and self.outputSelector.currentNode():
        # if self.FixedModelSelector.currentNode() and self.inputSelector.currentNode():

        if len(self.directory_path) > 0 and len(self.output_directory_path) > 0:
            self.computeButton.enabled = True           
        else:
            self.computeButton.enabled = False

    def tryint(self, s):
        # Used with alphanum_key() function to sort the files numerically
        try:
            return int(s)
        except:
            return s

    def alphanum_key(self, s):
        # For sorting of the filenames to be in numerical order 
        # Turn a string into a list of string and number chunks "z23a" -> ["z", 23, "a"]
        import re

        return [ self.tryint(c) for c in re.split('([0-9]+)', s) ]


    def onCompute(self):
        slicer.app.processEvents()

        import time
        # Show the status bar
        self.progressBar.show()

        # Initilize various parameters
        self.bone_labels = np.fromstring(self.lineedit.text, dtype=int, sep=',')

        # Check to see if self.bone_labels is set to negative on and if so set to the default labels
        if self.bone_labels[0] == -1:
            # Use the minimum and maximum of the Ref_Bone_Slider (which is based on the minimum and maximum label in the first image)
            self.bone_labels = range(int(self.Ref_Bone_Slider.minimum), int(self.Ref_Bone_Slider.maximum)+1)      

        self.max_bone_label = np.max(self.bone_labels)
        
        # Find all the files in the input folder
        self.files = os.listdir(self.directory_path)

        # Sort the files to be in order now
        self.files.sort(key=self.alphanum_key)

        if self.num_files > len(self.files):
            self.num_files = len(self.files)

        # Initilize a list of file indicies
        if self.num_files == -1:
            self.file_list = range(0, len(self.files))
        else:
            self.file_list = range(0, int(self.num_files))

        # Initilize Python list to hold all of the polydata
        # One index for each bone label (ranges from 1 to 9)
        polydata_list = []

        for i in range(0,self.max_bone_label+1):
            polydata_list.append([])

        # Load all of the images from the input folder
        images_list = []

        for curr_file in self.file_list:

            # Update the status bar
            self.progressBar.setValue(float(curr_file)/len(self.file_list)*100/10) # Use 10% of the status bar for loading the images
            slicer.app.processEvents()
            slicer.util.showStatusMessage("Loading Images...")

            if self.files[curr_file][-3:] == 'nii':
                print(str('Running file number ' + str(curr_file)))

                # Load the nifti (.nii) or analyze image (.img/.hdr) using SimpleITK               
                filename = os.path.join(self.directory_path, self.files[curr_file])
                image = self.load_image(filename)

                # It's upside-down when loaded, so add a flip filter
                if self.flip_image_vertically.checked == True:                    
                    imflip = vtk.vtkImageFlip()
                    try:
                        imflip.SetInputData(image.GetOutput())
                    except:
                        imflip.SetInputData(image)
                    imflip.SetFilteredAxis(1)
                    imflip.Update()
                    image = imflip.GetOutput()

                # If the images are flipped left/right so use a flip filter
                if self.flip_image_horizontally.checked == True:                   
                    imflip = vtk.vtkImageFlip()
                    try:
                        imflip.SetInputData(image.GetOutput())
                    except:
                        imflip.SetInputData(image)
                    imflip.SetFilteredAxis(0)
                    imflip.Update()
                    image = imflip.GetOutput()

                # Append the loaded image to the images_list
                images_list.append(image)

                # Temporarily push the image to Slicer (to have the correct class type for the model maker Slicer module)
                if self.debug_show_images.checked == True:                    
                    image = sitk.ReadImage(filename)
                    sitkUtils.PushToSlicer(image, str(curr_file), 0, overwrite=True)
                    node = slicer.util.getNode(str(curr_file))

        # If there is a non .nii file in the folder the images_list will be shorter than file_list
        # Redefine the file_list here
        self.file_list = range(0, len(images_list))

        iter = 0 # For status bar

        # Extract the surface from each image (i.e. the polydata)
        for label in self.bone_labels: 
            for curr_file in self.file_list:

                # Update the status bar (start at 10%)
                self.progressBar.setValue(10 + float(iter)/(len(self.bone_labels)*len(self.file_list))*100/5) # Use 20% of the bar for this
                slicer.app.processEvents()
                iter = iter + 1
                slicer.util.showStatusMessage("Extracting Surfaces...")

                polydata = self.Extract_Surface(images_list[curr_file], label)            
                polydata_list[label].append(polydata)       

                # Create model node ("Extract_Shapes") and add to scene
                if self.show_extracted_shapes.checked == True:                    
                    Extract_Shapes = slicer.vtkMRMLModelNode()
                    Extract_Shapes.SetAndObservePolyData(polydata)
                    modelDisplay = slicer.vtkMRMLModelDisplayNode()
                    modelDisplay.SetSliceIntersectionVisibility(True) # Show in slice view
                    modelDisplay.SetVisibility(True) # Show in 3D view
                    slicer.mrmlScene.AddNode(modelDisplay)
                    Extract_Shapes.SetAndObserveDisplayNodeID(modelDisplay.GetID())
                    slicer.mrmlScene.AddNode(Extract_Shapes) 

        # Save each registered bone separately?
        if self.Save_Extracted_Bones_Separately.checked == True:        

            for label in self.bone_labels: 
                for i in range(0,len(self.file_list)):           
                    path = os.path.join(self.output_directory_path, 'Bone_' + str(label) + '_position_' + str(i) + '.ply')    
                    plyWriter = vtk.vtkPLYWriter()
                    plyWriter.SetFileName(path)
                    plyWriter.SetInputData(polydata_list[label][i])
                    plyWriter.Write()
                    print('Saved: ' + path)

        # If this is set to true, stop the computation after extracting and smoothing the bones
        if self.Skip_Registration.checked == True:

            # Set the status bar to 100%
            self.progressBar.setValue(100)

            slicer.app.processEvents()
            slicer.util.showStatusMessage("Skipping Registration Step...")

            # Hide the status bar
            self.progressBar.hide() 

            # Reset the status message on bottom of 3D Slicer
            slicer.util.showStatusMessage(" ")     

            return 0;

        # Update the status bar (start at 30%)
        self.progressBar.setValue(30 + 20) # Use 20% of the bar for this
        slicer.app.processEvents()
        slicer.util.showStatusMessage("Calculating Reference Shapes...")

        # Don't use the mean shapes
        # Use the bone shapes from volunteer 1 instead of the mean shape for each bone

        # Initialize a Python list to hold the reference shapes
        reference_shapes = []

        # One index for each bone label (ranges from 1 to 9)
        for i in range(0,self.max_bone_label+1):
            reference_shapes.append([])

        # Get the bone shapes from the first volunteer and save them
        for label in self.bone_labels:
            reference_shapes[label].append(polydata_list[label][0]) 

        ######### Register the reference shape to each bone position #########

        iter_bar = 0 # For status bar

        for label in self.bone_labels:          
            for i in range(0,len(self.file_list)): 

                # Update the status bar (start at 50%)
                self.progressBar.setValue(50 + float(iter_bar)/(len(self.bone_labels)*len(self.file_list))*100/5) # Use 20% of the bar for this
                slicer.app.processEvents()
                iter_bar = iter_bar + 1
                slicer.util.showStatusMessage("Registering Reference Shapes...")
 
                transformedSource = self.IterativeClosestPoint(target=polydata_list[label][i], source=reference_shapes[label][0]) 

                # Replace the surface of file i and label with the registered reference shape for that particular bone
                polydata_list[label][i] = transformedSource

        iter_bar = 0 # For status bar

        ######### Register the reference bone (usually the radius for the wrist) for all the volunteers together #########
        for label in self.bone_labels:
            for i in range(0,len(self.file_list)): 

                # !! IMPORTANT !! Register the reference bone last (or else the bones afterwards will not register correctly)
                # Skip the reference label for now (register after all the other bones are registered)

                if label != self.ref_label:

                    # Update the status bar (start at 70%)
                    self.progressBar.setValue(70 + float(iter_bar)/(len(self.bone_labels)*len(self.file_list))*100/5) # Use 20% of the bar for this
                    slicer.app.processEvents()
                    iter_bar = iter_bar + 1
                    slicer.util.showStatusMessage("Registering Radius Together...")

                    polydata_list[label][i] = self.IterativeClosestPoint(target=polydata_list[self.ref_label][0], source=polydata_list[self.ref_label][i], reference=polydata_list[label][i]) 

        # Now register the reference label bones together
        for i in range(0,len(self.file_list)): 
            polydata_list[self.ref_label][i] = self.IterativeClosestPoint(target=polydata_list[self.ref_label][0], source=polydata_list[self.ref_label][i], reference=polydata_list[self.ref_label][i]) 


        ######### Save the output #########

        # Should the reference bone be remove (i.e. not saved) in the final PLY surface file?
        if self.Remove_Ref_Bone.checked == True:
            index = np.argwhere(self.bone_labels==self.ref_label)
            self.bone_labels = np.delete(self.bone_labels, index)

        # Combine the surfaces of the wrist for each person and save as a .PLY file
        for i in range(0,len(self.file_list)): 
            temp_combine_list = []

            # Update the status bar (start at 90%)
            self.progressBar.setValue(90 + float(i)/len(self.file_list)*100/10) # Use 10% of the bar for this
            slicer.app.processEvents()
            iter_bar = iter_bar + 1
            slicer.util.showStatusMessage("Saving Output Surfaces...")

            total_points = [0];

            # Get all the bones in a Python list for volunteer 'i'
            for label in self.bone_labels:
                temp_combine_list.append(polydata_list[label][i])

                # Print the number of surface points for this bone
                num_points = polydata_list[label][i].GetNumberOfPoints();
                print("Bone " + str(label) + " points: " + str(num_points)) 

                total_points.append(total_points[-1] + num_points)
            
            print("Total Cumulative Points" + str(total_points))

            # Combined the surfaces in the list into a single polydata surface
            combined_polydata = self.Combine_Surfaces(temp_combine_list)

            # Create model node ("ICP_Result") and add to scene
            if self.show_registered_shapes.checked == True:                
                ICP_Result = slicer.vtkMRMLModelNode()
                ICP_Result.SetAndObservePolyData(combined_polydata)
                modelDisplay = slicer.vtkMRMLModelDisplayNode()
                modelDisplay.SetSliceIntersectionVisibility(True) # Show in slice view
                modelDisplay.SetVisibility(True) # Show in 3D view
                slicer.mrmlScene.AddNode(modelDisplay)
                ICP_Result.SetAndObserveDisplayNodeID(modelDisplay.GetID())
                slicer.mrmlScene.AddNode(ICP_Result) 

            # Write the combined polydata surface to a .PLY file
            plyWriter = vtk.vtkPLYWriter()
            path = os.path.join(self.output_directory_path, str(self.files[i][:-4]) + '_combined.ply')
            plyWriter.SetFileName(path)
            plyWriter.SetInputData(combined_polydata)
            plyWriter.Write()
            print('Saved: ' + path)

            # Save each registered bone separately as well?
            if self.Save_Registered_Bones_Separately.checked == True:        

                for label in self.bone_labels:                   
                    path = os.path.join(self.output_directory_path, 'Position_' + str(i) + '_bone_' + str(label) + '.ply')    
                    plyWriter.SetFileName(path)
                    plyWriter.SetInputData(polydata_list[label][i])
                    plyWriter.Write()
                    print('Saved: ' + path)

        # Set the status bar to 100%
        self.progressBar.setValue(100)

        # Hide the status bar
        self.progressBar.hide() 

        # Reset the status message on bottom of 3D Slicer
        slicer.util.showStatusMessage(" ")        

        return 0

    def Extract_Surface(self, imgReader, label):
        # Extract a vtk surface from a vtk image

        treshold_filter = vtk.vtkImageThreshold()
        treshold_filter.ThresholdBetween(label,label)

        try:
            treshold_filter.SetInputConnection(imgReader.GetOutputPort())
        except:
            treshold_filter.SetInputData(imgReader)

        treshold_filter.ReplaceInOn()
        treshold_filter.SetInValue(255) # Replace the bone label = label to 255 (threshold again using contour filter)
        treshold_filter.Update()

        boneExtractor = vtk.vtkDiscreteMarchingCubes()

        try:
            boneExtractor.SetInputData(treshold_filter.GetOutput())
        except:
            boneExtractor.SetInputConnection(treshold_filter.GetOutputPort())

        # The bone label(s) of interest were set to 255 above using the image threshold filter
        boneExtractor.SetValue(0,255) 
        boneExtractor.Update()

        output_polydata = self.Smooth_Surface(boneExtractor.GetOutput())


        return output_polydata

    def IterativeClosestPoint(self, source, target, reference=[]):
        # Iterative closest point surface registration

        icp = vtk.vtkIterativeClosestPointTransform()

        try:
            icp.SetSource(source.GetPolyData())
            icp.SetTarget(target.GetPolyData())
        except:
            icp.SetSource(source)
            icp.SetTarget(target)

        if self.icp_mode == 'Rigid':
            icp.GetLandmarkTransform().SetModeToRigidBody()
        elif self.icp_mode == 'Similarity':
            icp.GetLandmarkTransform().SetModeToSimilarity()
        elif self.icp_mode == 'Affine':
            icp.GetLandmarkTransform().SetModeToAffine()
                
        # icp.DebugOn()
        icp.SetMaximumNumberOfIterations(int(self.IterationNumber))  
        icp.SetMaximumNumberOfLandmarks(int(self.LandmarkNumber))  
        icp.SetMaximumMeanDistance(self.RMS_Number) # A small number would use the maximum number of iterations
        icp.StartByMatchingCentroidsOn()

        # RMS mode is the square root of the average of the sum of squares of the closest point distances
        icp.SetMeanDistanceModeToRMS()

        icp.Modified()
        icp.Update()

        # Apply the resulting transform to the vtk poly data 
        icpTransformFilter = vtk.vtkTransformPolyDataFilter()
        try:
            icpTransformFilter.SetInputData(source.GetPolyData())
        except:
            icpTransformFilter.SetInputData(source)

        icpTransformFilter.SetTransform(icp)
        icpTransformFilter.Update()

        # Update the source object with the transformed object
        transformedSource = icpTransformFilter.GetOutput()

        # If there is a reference surface apply the ICP transform to it
        if reference != []:

            # Apply the resulting transform also to the reference surface vtk poly data 
            icpTransformFilter = vtk.vtkTransformPolyDataFilter()
            try:
                icpTransformFilter.SetInputData(reference.GetPolyData())
            except:
                icpTransformFilter.SetInputData(reference)

            icpTransformFilter.SetTransform(icp)
            icpTransformFilter.Update()

            # Update the source object with the transformed object
            transformedReference = icpTransformFilter.GetOutput()

            # Is there is a referece surface return that instead
            return transformedReference
        else:
            # If there in NOT a reference surface, return the registered source
            return transformedSource

    def Smooth_Surface(self, surface):
        # Take a vtk surface and run it through the smoothing pipeline

        boneNormals = vtk.vtkPolyDataNormals()
        try:
            boneNormals.SetInputData(surface)
        except:
            boneNormals.SetInputConnection(surface.GetOutputPort())

        boneNormals.Update()

        # Clean the polydata so that the edges are shared!
        cleanPolyData = vtk.vtkCleanPolyData()
        cleanPolyData.SetInputConnection(boneNormals.GetOutputPort())
        cleanPolyData.Update()
        polydata = cleanPolyData.GetOutput()

        if self.smoothing_iterations > 0:
            # Apply laplacian smoothing to the surface
            smoothingFilter = vtk.vtkSmoothPolyDataFilter()
            smoothingFilter.SetInputData(polydata)
            smoothingFilter.SetNumberOfIterations(int(self.smoothing_iterations))
            smoothingFilter.SetRelaxationFactor(self.relaxation_factor)
            smoothingFilter.Update()

            polydata = smoothingFilter.GetOutput()

        if self.decimate_surface  > 0 and self.decimate_surface < 1:
            # We want to preserve topology (not let any cracks form). This may
            # limit the total reduction possible, which we have specified at 80%.
            deci = vtk.vtkDecimatePro()
            deci.SetInputData(polydata)
            deci.SetTargetReduction(self.decimate_surface)
            deci.PreserveTopologyOn()
            deci.Update()
            polydata = deci.GetOutput()

        # Clean the polydata so that the edges are shared!
        cleanPolyData = vtk.vtkCleanPolyData()
        cleanPolyData.SetInputData(polydata)
        cleanPolyData.Update()
        polydata = cleanPolyData.GetOutput()

        # Generate surface normals to give a better visualization        
        normals = vtk.vtkPolyDataNormals()
        normals.SetInputData(polydata)
        normals.Update()
        polydata = normals.GetOutput()

        return polydata

    def load_image(self, ImgFileName):
        # Load an image using the vtk image reader 

        imgReader = vtk.vtkNIFTIImageReader()
        imgReader.SetFileName(ImgFileName)
        imgReader.Update()

        # This is needed for working on the Mac
        # Perhaps the imgReader don't finish running in time?
        # VTK uses 'lazy functions'
        from vtk.util.numpy_support import vtk_to_numpy
        array = vtk_to_numpy(imgReader.GetOutput().GetPointData().GetScalars())
        
        return imgReader
    
    def Combine_Surfaces(self, polydata_list):
        # Combine some number of polydata together
        # For example, useful for combining all the bones of a joint back together for saving to a .PLY file

        # Append the meshes together
        appendFilter = vtk.vtkAppendPolyData()

        # Loop through each surface in the list of polydata and input into the filter
        for i in range(0, len(polydata_list)):
            appendFilter.AddInputData(polydata_list[i])

        # Update the combinded polydata filter
        appendFilter.Update()

        # Get the output of the filter (i.e. combined_polydata holds the combined surfaces)
        combined_polydata = appendFilter.GetOutput()
    
        return combined_polydata

if __name__ == "__main__":
    # TODO: need a way to access and parse command line arguments
    # TODO: ideally command line args should handle --xml

    import sys
    print(sys.argv)
    print("Running this module from the command line is not supported yet.")
