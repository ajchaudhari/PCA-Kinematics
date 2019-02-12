from __main__ import vtk, qt, ctk, slicer
import EditorLib

import SimpleITK as sitk
import sitkUtils
import numpy as np
import multiprocessing
import timeit
import time
import os

from vtk.util.numpy_support import vtk_to_numpy

#
# Create_PCA_Model
#
class PCA_Kinematics:
    def __init__(self, parent):
        import string
        parent.title = "Wrist PCA-Kinematics"
        parent.categories = ["PCA-Kinematics"]
        parent.contributors = ["Brent Foster (UC Davis)"]
        parent.helpText = string.Template("""
        
        PURPOSE: Use this 3D Slicer module to construct the bone displacement model using a principal component analysis (PCA) based approach as described in Foster et al Journal of Biomechanics (https://doi.org/10.1016/j.jbiomech.2019.01.030). 
        <br>
        <br>
        INPUT: A folder of training data (.ply surfaces) created using the "Create PCA Kinematics Training Data" module. 
        The training data must be surfaces in correspondence and have the same bone shape (but in different positions and orientations)
        for the corresponding bones. 
        <br>
        <br>
        OUTPUT: The displacement model will appear in the 3D view within 3D Slicer. There are various functions to interact with the model. 
        <br>
        <br>
        FUNCTIONS: The module has the following functions
        <br>
        (1) 5 sliders: Allows the user to vary the model coefficient of the first five basis functions (i.e. eigenvectors). This will change the displacement model and the rendering will be updated. 
        <br>
        (2) Create PCA Kinematic Model: Using the folder of training data to construct the displacement model. 
        <br>
        (3) Model Coefficient Fitting and Interpolation: Using the constructed displacement model, fit the model to each surface within a folder to find the best fitting model coefficients which is then saved to a text file. Note that these surfaces must be in correspondence with the displacement model.
        <br>
        (4) Rendering Options: Several rendering options are provided.

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
        parent.index = 1
        self.parent = parent

class PCA_KinematicsWidget:
    def __init__(self, parent=None):
        self.parent = parent
        self.logic = None
        self.ImageNode = None

        # Font settings for the module
        self.font_type = "Arial"
        self.font_size = 12

        # Number of eigenvalues to use
        self.NumEVs = 5

        # Create a variable to hold the model
        # This will be checked to see if the model needs to be run again
        self.pca_model = []
        self.polydata_list = []

        # Flag to see if a new folder of surfaces was selected
        self.New_Folder_Selected = False

        # Flag to show a new transform
        self.Show_New_Transform = False

        # Flag that a new transform was selected 
        self.New_Transform_Selected = False

        # Set the default maximum/minimum value for the sliders
        self.slider_range = 1

        # Flag to check if OnCompute() is already running
        self.Already_Running = False

        # Variable to hold the landmark indicies for the wrist angle measurements
        self.landmark_index = []

        # Variable to hold the landmark indicies for the bone distance measurements
        self.distance_landmark_index = []

        # Variable to hold a list of volunteer numbers if only using a subset (as defined in the debug collaspse form)
        self.volunteer_selected = []

        # Flag to show the progress bar or not
        self.show_progress_bar = True

        # Initilize a flag to know if the sliders are reseting
        # This prevents running the autorun for each slider that is being reset
        self.reseting_state = False
        
        # Variables to hold the file paths 
        self.directory_path = []
        self.Directory_Input_Surfaces_Fitting = []
        self.fitting_output_directory_path = []

    def setup(self):
        frame = qt.QFrame()
        frameLayout = qt.QFormLayout()
        frame.setLayout(frameLayout)
        self.parent.layout().addWidget(frame)

        # Choose Directory button to choose the folder for saving the registered image 
        self.directoryButton = qt.QPushButton("Choose Training Data Folder")
        self.directoryButton.setFont(qt.QFont(self.font_type, self.font_size))
        self.directoryButton.toolTip = "Choose the folder with the training data surfaces (in .PLY format). \
        This training data is easily created using the 'Create PCA Kinematics Training Data' module."
        frameLayout.addWidget(self.directoryButton)
        self.directoryButton.connect('clicked()', self.onDirectoryButtonClick)

        # Compute the bone displacement model button
        self.computeButton = qt.QPushButton("Create Bone Displacement Model")
        self.computeButton.setFont(qt.QFont(self.font_type, self.font_size))
        self.computeButton.toolTip = "Create the statistical model of bone displacement using the folder of training data."
        frameLayout.addWidget(self.computeButton)
        self.computeButton.connect('clicked()', self.onCompute)

        # Model coefficient selection sliders collapsible button
        self.Model_Sliders_CollapsibleButton = ctk.ctkCollapsibleButton()
        self.Model_Sliders_CollapsibleButton.setFont(qt.QFont(self.font_type, self.font_size))
        self.Model_Sliders_CollapsibleButton.text = "Model Coefficent Select"
        self.Model_Sliders_CollapsibleButton.collapsed = True # Default is to not show     
        frameLayout.addWidget(self.Model_Sliders_CollapsibleButton) 

        # Layout within the model fitting collapsible button
        self.Model_Sliders_FormLayout = qt.QFormLayout(self.Model_Sliders_CollapsibleButton)

        # Slider for selecting the first model coefficient     
        self.label = qt.QLabel()
        self.label.setFont(qt.QFont(self.font_type, self.font_size))
        self.label.setText("1st Coefficent: ")
        tooltip = "Select the 1st model coefficient (i.e. the scaling term (alpha)) to multiply times the first eigenvalue."
        self.label.setToolTip(tooltip)
        self.FirstEVSlider = ctk.ctkSliderWidget()
        self.FirstEVSlider.setFont(qt.QFont(self.font_type, self.font_size))
        self.FirstEVSlider.setToolTip(tooltip)
        self.FirstEVSlider.minimum = -self.slider_range
        self.FirstEVSlider.maximum = self.slider_range
        self.FirstEVSlider.value = 0
        self.FirstEVSlider.singleStep = 0.1
        self.FirstEVSlider.tickInterval = 0.1
        self.FirstEVSlider.decimals = 2
        self.FirstEVSlider.connect('valueChanged(double)', self.onFirstEVSliderChange)
        self.Model_Sliders_FormLayout.addRow(self.label, self.FirstEVSlider)        
        self.FirstEV = self.FirstEVSlider.value # Set default value

        # Slider for selecting the scaling of the second eigenvector of the model  
        self.label = qt.QLabel()
        self.label.setFont(qt.QFont(self.font_type, self.font_size))
        self.label.setText("2nd Coefficent: ")
        tooltip = "Select the 2nd model coefficient (i.e. the scaling term (alpha)) to multiply times the first eigenvalue."
        self.label.setToolTip(tooltip)
        self.SecondEVSlider = ctk.ctkSliderWidget()
        self.SecondEVSlider.setFont(qt.QFont(self.font_type, self.font_size))
        self.SecondEVSlider.setToolTip(tooltip)
        self.SecondEVSlider.minimum = -self.slider_range
        self.SecondEVSlider.maximum = self.slider_range
        self.SecondEVSlider.value = 0
        self.SecondEVSlider.singleStep = 0.05
        self.SecondEVSlider.tickInterval = 0.1
        self.SecondEVSlider.decimals = 2
        self.SecondEVSlider.connect('valueChanged(double)', self.onSecondEVSliderChange)
        self.Model_Sliders_FormLayout.addRow(self.label, self.SecondEVSlider)        
        self.SecondEV = self.SecondEVSlider.value # Set default value

        # Slider for selecting the scaling of the third eigenvector of the model    
        self.label = qt.QLabel()
        self.label.setFont(qt.QFont(self.font_type, self.font_size))
        self.label.setText("3rd Coefficent: ")
        tooltip = "Select the 3rd model coefficient (i.e. the scaling term (alpha)) to multiply times the first eigenvalue."
        self.label.setToolTip(tooltip)
        self.ThirdEVSlider = ctk.ctkSliderWidget()
        self.ThirdEVSlider.setFont(qt.QFont(self.font_type, self.font_size))
        self.ThirdEVSlider.setToolTip(tooltip)
        self.ThirdEVSlider.minimum = -self.slider_range
        self.ThirdEVSlider.maximum = self.slider_range
        self.ThirdEVSlider.value = 0
        self.ThirdEVSlider.singleStep = 0.05
        self.ThirdEVSlider.tickInterval = 0.1
        self.ThirdEVSlider.decimals = 2
        self.ThirdEVSlider.connect('valueChanged(double)', self.onThirdEVSliderChange)
        self.Model_Sliders_FormLayout.addRow(self.label, self.ThirdEVSlider)
        self.ThirdEV = self.ThirdEVSlider.value # Set default value

        # Slider for selecting the scaling of the fourth eigenvector of the model 
        self.label = qt.QLabel()
        self.label.setFont(qt.QFont(self.font_type, self.font_size))
        self.label.setText("4th Coefficent: ")
        tooltip = "Select the 4th model coefficient (i.e. the scaling term (alpha)) to multiply times the first eigenvalue."
        self.label.setToolTip(tooltip)   
        self.FourthEVSlider = ctk.ctkSliderWidget()
        self.FourthEVSlider.setFont(qt.QFont(self.font_type, self.font_size))
        self.FourthEVSlider.setToolTip(tooltip)
        self.FourthEVSlider.minimum = -self.slider_range
        self.FourthEVSlider.maximum = self.slider_range
        self.FourthEVSlider.value = 0
        self.FourthEVSlider.singleStep = 0.05
        self.FourthEVSlider.tickInterval = 0.1
        self.FourthEVSlider.decimals = 2
        self.FourthEVSlider.connect('valueChanged(double)', self.onFourthEVSliderChange)
        self.Model_Sliders_FormLayout.addRow(self.label, self.FourthEVSlider)        
        self.FourthEV = self.FourthEVSlider.value # Set default value

        # Slider for selecting the scaling of the fifth eigenvector of the model    
        self.label = qt.QLabel()
        self.label.setFont(qt.QFont(self.font_type, self.font_size))
        self.label.setText("5th Coefficent: ")
        tooltip = "Select the 5th model coefficient (i.e. the scaling term (alpha)) to multiply times the first eigenvalue."
        self.label.setToolTip(tooltip)        
        self.FifthEVSlider = ctk.ctkSliderWidget()
        self.FifthEVSlider.setFont(qt.QFont(self.font_type, self.font_size))
        self.FifthEVSlider.setToolTip(tooltip)
        self.FifthEVSlider.minimum = -self.slider_range
        self.FifthEVSlider.maximum = self.slider_range
        self.FifthEVSlider.value = 0
        self.FifthEVSlider.singleStep = 0.05
        self.FifthEVSlider.tickInterval = 0.1
        self.FifthEVSlider.decimals = 2
        self.FifthEVSlider.connect('valueChanged(double)', self.onFifthEVSliderChange)
        self.Model_Sliders_FormLayout.addRow(self.label, self.FifthEVSlider)        
        self.FifthEV = self.FifthEVSlider.value # Set default value
        
        # Reset model coefficient sliders button
        self.resetButton = qt.QPushButton("Reset Model Coefficient Sliders")
        self.resetButton.setFont(qt.QFont(self.font_type, self.font_size))
        self.resetButton.toolTip = "Reset all the model coefficient selection sliders back to all zeros."
        self.Model_Sliders_FormLayout.addWidget(self.resetButton)
        self.resetButton.connect('clicked()', self.onResetButton)

        # Model Coefficent Fitting and Interpolation Collapse button
        self.Model_Fitting_CollapsibleButton = ctk.ctkCollapsibleButton()
        self.Model_Fitting_CollapsibleButton.setFont(qt.QFont(self.font_type, self.font_size))
        self.Model_Fitting_CollapsibleButton.text = "Model Coefficent Fitting and Interpolation"
        self.Model_Fitting_CollapsibleButton.collapsed = True # Default is to not show     
        frameLayout.addWidget(self.Model_Fitting_CollapsibleButton) 

        # Layout within the model fitting collapsible button
        self.Eigenvalue_Fitting_FormLayout = qt.QFormLayout(self.Model_Fitting_CollapsibleButton)

        # Choose a folder of surfaces in correspondence to fit the model to
        self.label = qt.QLabel()
        self.label.setFont(qt.QFont(self.font_type, self.font_size))
        self.label.setText("Surfaces to Fit: ")
        tooltip = "Fit the constructed bone displacement model to each surface within a folder of surfaces. Note that these surfaces must be in correspondence with the bone displacement model surface."
        self.label.setToolTip(tooltip)   
        self.directorySurfaceFittingButton = qt.QPushButton("Choose Input Folder")
        self.directorySurfaceFittingButton.setFont(qt.QFont(self.font_type, self.font_size))
        self.directorySurfaceFittingButton.setToolTip(tooltip)
        self.Eigenvalue_Fitting_FormLayout.addRow(self.label, self.directorySurfaceFittingButton)      
        self.directorySurfaceFittingButton.connect('clicked()', self.onDirectorySurfaceFittingButtonClick)

        # Choose a folder to output the resulting text file to (containing the fitted model coefficients)
        self.label = qt.QLabel()
        self.label.setFont(qt.QFont(self.font_type, self.font_size))
        self.label.setText("Output Folder: ")
        tooltip = "Choose a folder to save the resulting text files to (which contain a list of the filenames fitted and the resulting model coefficients)."
        self.label.setToolTip(tooltip)
        self.directoryFittingOutputButton = qt.QPushButton("Choose Output Folder")
        self.directoryFittingOutputButton.setFont(qt.QFont(self.font_type, self.font_size))
        self.directoryFittingOutputButton.setToolTip(tooltip)
        self.Eigenvalue_Fitting_FormLayout.addRow(self.label, self.directoryFittingOutputButton)        
        self.directoryFittingOutputButton.connect('clicked()', self.onDirectoryFittingOutputButtonClick)           

        # Button to run the fitting procedure on the surfaces in correspondence
        self.PCAFittingButton = qt.QPushButton("Run Fitting Procedure")
        self.PCAFittingButton.setFont(qt.QFont(self.font_type, self.font_size))
        self.PCAFittingButton.toolTip = "Find the eigenvalue coefficients of the surfaces in correspondence \
        which are located in the above folder. Then save the coefficients to a text file."
        self.Eigenvalue_Fitting_FormLayout.addWidget(self.PCAFittingButton)
        self.PCAFittingButton.connect('clicked()', self.onPCAFittingButtonClicked)

        # Slider to vary TIME instead of eigenvalue scaling on the fitted coefficients
        self.label = qt.QLabel()
        self.label.setFont(qt.QFont(self.font_type, self.font_size))
        self.label.setText("Time Slider:")
        tooltip = "Fit a curve to the eigenvalue scaling at the positions in the above folder and select the time point by varying multiple model coefficients at once."
        self.label.setToolTip(tooltip)
        self.TimeSelectSlider_Fitted = ctk.ctkSliderWidget()
        self.TimeSelectSlider_Fitted.setFont(qt.QFont(self.font_type, self.font_size))
        self.TimeSelectSlider_Fitted.setToolTip(tooltip)
        self.TimeSelectSlider_Fitted.minimum = -1
        self.TimeSelectSlider_Fitted.maximum = 1
        self.TimeSelectSlider_Fitted.value = 0
        self.TimeSelectSlider_Fitted.singleStep = 0.1
        self.TimeSelectSlider_Fitted.tickInterval = 0.1
        self.TimeSelectSlider_Fitted.decimals = 2
        self.TimeSelectSlider_Fitted.connect('valueChanged(double)', self.onTimeSelectSlider_FittedChange)
        self.Eigenvalue_Fitting_FormLayout.addRow(self.label, self.TimeSelectSlider_Fitted)        
        self.TimeSelected_Fitted = self.TimeSelectSlider_Fitted.value # Set default value

        # Slider to select the polynomial order of the fitting of the eigenvalues
        # This is the second slider of this type and is used for interpolating between
        # The positions in the above folder (as opposed to using the table of values)
        self.label = qt.QLabel()
        self.label.setFont(qt.QFont(self.font_type, self.font_size))
        self.label.setText("Fitting Order:")
        tooltip = "Choose the order of the fitting. 1 = linear, 2 = 2nd order, etc."
        self.label.setToolTip(tooltip)
        self.FittingOrderSlider_Fitted = ctk.ctkSliderWidget()
        self.FittingOrderSlider_Fitted.setFont(qt.QFont(self.font_type, self.font_size))
        self.FittingOrderSlider_Fitted.setToolTip(tooltip)
        self.FittingOrderSlider_Fitted.minimum = 1
        self.FittingOrderSlider_Fitted.maximum = 5
        self.FittingOrderSlider_Fitted.value = 1
        self.FittingOrderSlider_Fitted.singleStep = 1
        self.FittingOrderSlider_Fitted.tickInterval = 1
        self.FittingOrderSlider_Fitted.decimals = 0
        self.FittingOrderSlider_Fitted.connect('valueChanged(double)', self.onFittingOrderSlider_FittedChange)
        self.Eigenvalue_Fitting_FormLayout.addRow(self.label, self.FittingOrderSlider_Fitted)        
        self.FittingOrder_Fitted = self.FittingOrderSlider_Fitted.value # Set default value

        # Rendering Options Collapse button
        self.Rendering_Options_CollapsibleButton = ctk.ctkCollapsibleButton()
        self.Rendering_Options_CollapsibleButton.setFont(qt.QFont(self.font_type, self.font_size))
        self.Rendering_Options_CollapsibleButton.text = "Rendering Options"
        self.Rendering_Options_CollapsibleButton.collapsed = True # Default is to not show  
        frameLayout.addWidget(self.Rendering_Options_CollapsibleButton) 

        # Layout within the model fitting collapsible button
        self.Rendering_Options_FormLayout = qt.QFormLayout(self.Rendering_Options_CollapsibleButton)

        # Slider for selecting the amount of red in the rendered surface   
        self.label = qt.QLabel()
        self.label.setFont(qt.QFont(self.font_type, self.font_size))
        self.label.setText("Red: ")
        tooltip = "Select the amount of red to have in the rendered surface."
        self.label.setToolTip(tooltip)
        self.RedColorSlider = ctk.ctkSliderWidget()
        self.RedColorSlider.setFont(qt.QFont(self.font_type, self.font_size))
        self.RedColorSlider.setToolTip(tooltip)
        self.RedColorSlider.minimum = 0
        self.RedColorSlider.maximum = 1
        self.RedColorSlider.value = 0.8
        self.RedColorSlider.singleStep = 0.01
        self.RedColorSlider.tickInterval = 0.01
        self.RedColorSlider.decimals = 2
        self.RedColorSlider.connect('valueChanged(double)', self.onRedColorSliderChange)
        self.Rendering_Options_FormLayout.addRow(self.label, self.RedColorSlider)        
        self.RedColor = self.RedColorSlider.value # Set default value

        # Slider for selecting the amount of green in the rendered surface     
        self.label = qt.QLabel()
        self.label.setFont(qt.QFont(self.font_type, self.font_size))
        self.label.setText("Green: ")
        tooltip = "Select the amount of green to have in the rendered surface."
        self.label.setToolTip(tooltip)
        self.GreenColorSlider = ctk.ctkSliderWidget()
        self.GreenColorSlider.setFont(qt.QFont(self.font_type, self.font_size))
        self.GreenColorSlider.setToolTip(tooltip)
        self.GreenColorSlider.minimum = 0
        self.GreenColorSlider.maximum = 1
        self.GreenColorSlider.value = 0.8
        self.GreenColorSlider.singleStep = 0.01
        self.GreenColorSlider.tickInterval = 0.01
        self.GreenColorSlider.decimals = 2
        self.GreenColorSlider.connect('valueChanged(double)', self.onGreenColorSliderChange)
        self.Rendering_Options_FormLayout.addRow(self.label, self.GreenColorSlider)        
        self.GreenColor = self.GreenColorSlider.value # Set default value

        # Slider for selecting the amount of blue in the rendered surface   
        self.label = qt.QLabel()
        self.label.setFont(qt.QFont(self.font_type, self.font_size))
        self.label.setText("Blue: ")
        tooltip = "Select the amount of blue to have in the rendered surface."
        self.label.setToolTip(tooltip)
        self.BlueColorSlider = ctk.ctkSliderWidget()
        self.BlueColorSlider.setFont(qt.QFont(self.font_type, self.font_size))
        self.BlueColorSlider.setToolTip(tooltip)
        self.BlueColorSlider.minimum = 0
        self.BlueColorSlider.maximum = 1
        self.BlueColorSlider.value = 0.8
        self.BlueColorSlider.singleStep = 0.01
        self.BlueColorSlider.tickInterval = 0.01
        self.BlueColorSlider.decimals = 2
        self.BlueColorSlider.connect('valueChanged(double)', self.onBlueColorSliderChange)
        self.Rendering_Options_FormLayout.addRow(self.label, self.BlueColorSlider)        
        self.BlueColor = self.BlueColorSlider.value # Set default value

        # Render the surface as points
        self.represent_points = qt.QCheckBox("Render as points")
        self.represent_points.setFont(qt.QFont(self.font_type, self.font_size))
        self.represent_points.toolTip = "When checked, the surface will be rendered using the points option."
        self.represent_points.checked = False
        self.Rendering_Options_FormLayout.addWidget(self.represent_points) 
        self.represent_points.connect('clicked()', self.onCompute)

        # Render the surface as a wireframe
        self.represent_wireframe = qt.QCheckBox("Render as wireframe")
        self.represent_wireframe.setFont(qt.QFont(self.font_type, self.font_size))
        self.represent_wireframe.toolTip = "When checked, the surface will be rendered using the wireframe option."
        self.represent_wireframe.checked = False
        self.Rendering_Options_FormLayout.addWidget(self.represent_wireframe) 
        self.represent_wireframe.connect('clicked()', self.onCompute)

        # Render the surface as a surface
        self.represent_surface = qt.QCheckBox("Render as surface")
        self.represent_surface.setFont(qt.QFont(self.font_type, self.font_size))
        self.represent_surface.toolTip = "When checked, the surface will be rendered using the surface option."
        self.represent_surface.checked = True
        self.Rendering_Options_FormLayout.addWidget(self.represent_surface) 
        self.represent_surface.connect('clicked()', self.onCompute)

        # Render the surface as a surface with edges
        self.represent_surface_edges = qt.QCheckBox("Render as surface with edges")
        self.represent_surface_edges.setFont(qt.QFont(self.font_type, self.font_size))
        self.represent_surface_edges.toolTip = "When checked, the surface will be rendered using the surface with edges option."
        self.represent_surface_edges.checked = False
        self.Rendering_Options_FormLayout.addWidget(self.represent_surface_edges) 
        self.represent_surface_edges.connect('clicked()', self.onCompute)

        # Show normal vector checkmark
        self.show_glyph = qt.QCheckBox("Show Surface Normal Vectors")
        self.show_glyph.setFont(qt.QFont(self.font_type, self.font_size))
        self.show_glyph.toolTip = "When checked, the normal vector at each point on the surface will be rendered. This is useful for debugging surface mesh issues."
        self.show_glyph.checked = False
        self.Rendering_Options_FormLayout.addWidget(self.show_glyph) 
        self.show_glyph.connect('clicked()', self.onCompute)

        # Auto delete the last model 
        self.auto_delete = qt.QCheckBox("Auto Delete")
        self.auto_delete.setFont(qt.QFont(self.font_type, self.font_size))
        self.auto_delete.toolTip = "When checked, auto delete the surface generated last time. This will delete all the models whose name begin with PCA"
        self.auto_delete.checked = True
        self.Rendering_Options_FormLayout.addWidget(self.auto_delete) 

        # Button for looping through the model coefficients (for visualizing)
        self.LoopCoefficientsButton = qt.QPushButton("Loop Model Coefficients")
        self.LoopCoefficientsButton.setFont(qt.QFont(self.font_type, self.font_size))
        self.LoopCoefficientsButton.toolTip = "Press this button to loop through the model coefficients one at a time. Useful for rendering what the model looks like."
        self.Rendering_Options_FormLayout.addWidget(self.LoopCoefficientsButton)
        self.LoopCoefficientsButton.connect('clicked()', self.onLoopCoefficientsButton)

        # Slider for selecting step size for the model coefficient looping 
        self.label = qt.QLabel()
        self.label.setFont(qt.QFont(self.font_type, self.font_size))
        self.label.setText("Coefficent Looping Step Size: ")
        tooltip = "Select the step size for the coefficient looping. A smaller step size will take longer but will be smoother."
        self.label.setToolTip(tooltip)
        self.LoopStepSizeSlider = ctk.ctkSliderWidget()
        self.LoopStepSizeSlider.setFont(qt.QFont(self.font_type, self.font_size))
        self.LoopStepSizeSlider.setToolTip(tooltip)
        self.LoopStepSizeSlider.minimum = 1
        self.LoopStepSizeSlider.maximum = 50
        self.LoopStepSizeSlider.value = 25
        self.LoopStepSizeSlider.singleStep = 1
        self.LoopStepSizeSlider.tickInterval = 1
        self.LoopStepSizeSlider.decimals = 0
        self.LoopStepSizeSlider.connect('valueChanged(double)', self.onLoopStepSizeSliderChange)
        self.Rendering_Options_FormLayout.addRow(self.label, self.LoopStepSizeSlider)        
        self.loop_coefficients_num_steps = self.LoopStepSizeSlider.value # Set default value

        # Debug Options Collapse button
        self.Debug_Options_CollapsibleButton = ctk.ctkCollapsibleButton()
        self.Debug_Options_CollapsibleButton.setFont(qt.QFont(self.font_type, self.font_size))
        self.Debug_Options_CollapsibleButton.text = "Debug"
        self.Debug_Options_CollapsibleButton.collapsed = True # Default is to not show     
        frameLayout.addWidget(self.Debug_Options_CollapsibleButton) 

        # Layout within the model fitting collapsible button
        self.Debug_Options_FormLayout = qt.QFormLayout(self.Debug_Options_CollapsibleButton)
      
        # Slider for selecting model coefficient slider range     
        self.label = qt.QLabel()
        self.label.setFont(qt.QFont(self.font_type, self.font_size))
        self.label.setText("Select Coefficent Slider Range: ")
        tooltip = "Select the range of the model coefficient sliders (i.e. the 5 sliders at the top of the module)."
        self.label.setToolTip(tooltip)
        self.CoefficentRangeSlider = ctk.ctkSliderWidget()
        self.CoefficentRangeSlider.setFont(qt.QFont(self.font_type, self.font_size))
        self.CoefficentRangeSlider.setToolTip(tooltip)
        self.CoefficentRangeSlider.minimum = 0.1
        self.CoefficentRangeSlider.maximum = 10
        self.CoefficentRangeSlider.value = 1
        self.CoefficentRangeSlider.singleStep = 0.1
        self.CoefficentRangeSlider.tickInterval = 0.05
        self.CoefficentRangeSlider.decimals = 2
        self.CoefficentRangeSlider.connect('valueChanged(double)', self.onCoefficentRangeSliderChange)
        self.Debug_Options_FormLayout.addRow(self.label, self.CoefficentRangeSlider)        
        self.slider_range = self.CoefficentRangeSlider.value # Set default value

        # Progress Bar (so the user knows how much longer the computation will likely take)
        self.progressBar = qt.QProgressBar()
        self.progressBar.setFont(qt.QFont(self.font_type, self.font_size))
        self.progressBar.setValue(0)
        frameLayout.addWidget(self.progressBar)
        self.progressBar.hide()  

    def onCoefficentRangeSliderChange(self, newValue):
        # Save the new value from this slider
        self.slider_range = newValue;

        # Update the five model coefficient sliders with the new range
        self.FirstEVSlider.minimum  = -self.slider_range
        self.FirstEVSlider.maximum  = self.slider_range
        self.SecondEVSlider.minimum = -self.slider_range
        self.SecondEVSlider.maximum = self.slider_range
        self.ThirdEVSlider.minimum  = -self.slider_range
        self.ThirdEVSlider.maximum  = self.slider_range
        self.FourthEVSlider.minimum = -self.slider_range
        self.FourthEVSlider.maximum = self.slider_range
        self.FifthEVSlider.minimum  = -self.slider_range
        self.FifthEVSlider.maximum  = self.slider_range

    def onLoopStepSizeSliderChange(self, newValue):
        # Save the new value from this slider
        self.loop_coefficients_num_steps = newValue;

    def onRedColorSliderChange(self, newValue):
        self.RedColor = newValue

        # Run the onCompute function to update the current rendering
        self.onCompute()

    def onGreenColorSliderChange(self, newValue):
        self.GreenColor = newValue

        # Run the onCompute function to update the current rendering
        self.onCompute()

    def onBlueColorSliderChange(self, newValue):
        self.BlueColor = newValue

        # Run the onCompute function to update the current rendering
        self.onCompute()

    def CreateGlyphs(self, polydata, flipNormals):
        # Create vectors normal to the surface at each point
        # The vectors are called Glyphs
        # To flip the orientation set flipNormals to True or False

        # Compute the normal to the polydata surface
        normals = vtk.vtkPolyDataNormals()
        normals.SetInputData(polydata)
        normals.FlipNormalsOn()
        normals.SetFeatureAngle(60.0)
        normals.Update()
        polydata = normals.GetOutput()

        # vtkReverseSense is used to flip the vector 
        reverse = vtk.vtkReverseSense()

        # Choose a random subset of points.
        maskPts = vtk.vtkMaskPoints()
        maskPts.SetOnRatio(1)
        maskPts.RandomModeOn()

        if flipNormals:
            reverse.SetInputData(polydata)
            reverse.ReverseCellsOn()
            reverse.ReverseNormalsOn()
            maskPts.SetInputConnection(reverse.GetOutputPort())
        else:
            maskPts.SetInputData(polydata)

        # Source for the glyph filter
        arrow = vtk.vtkArrowSource()
        arrow.SetTipResolution(4) # 16
        arrow.SetTipLength(0.05) # 0.3
        arrow.SetTipRadius(0.05) # 0.1

        glyph = vtk.vtkGlyph3D()
        glyph.SetSourceConnection(arrow.GetOutputPort())
        glyph.SetInputConnection(maskPts.GetOutputPort())
        glyph.SetVectorModeToUseNormal()
        glyph.SetScaleFactor(1) # 1
        glyph.SetColorModeToColorByVector()
        glyph.SetScaleModeToScaleByVector()
        glyph.OrientOn()
        glyph.Update()

        return glyph


    def onResetButton(self):
        # Reset all the model coefficient selection sliders back to zero
        
        # Prevent the autorun (since the sliders are resetting)
        self.reseting_state = True

        # Reset all the sliders back to zero
        self.FirstEVSlider.value  = 0
        self.SecondEVSlider.value = 0
        self.ThirdEVSlider.value  = 0
        self.FourthEVSlider.value = 0
        self.FifthEVSlider.value  = 0

        # Set the reseting flag back to false now (if needed)
        self.reseting_state = False

        # Rebuild the model and create a new instance using the new EVs
        self.onCompute()

    def onFirstEVSliderChange(self, newValue):
        self.FirstEV = newValue   

        if self.reseting_state == False:              
            # Rebuild the model and create a new instance using the new EV
            self.onCompute()

    def onSecondEVSliderChange(self, newValue):
        self.SecondEV = newValue   

        if self.reseting_state == False:
            # Rebuild the model and create a new instance using the new EV
            self.onCompute()

    def onThirdEVSliderChange(self, newValue):
        self.ThirdEV = newValue 

        if self.reseting_state == False:
            # Rebuild the model and create a new instance using the new EV
            self.onCompute()

    def onFourthEVSliderChange(self, newValue):
        self.FourthEV = newValue 

        if self.reseting_state == False:
            # Rebuild the model and create a new instance using the new EV
            self.onCompute()

    def onFifthEVSliderChange(self, newValue):
        self.FifthEV = newValue 

        if self.reseting_state == False:
            # Rebuild the model and create a new instance using the new EV
            self.onCompute()

    def onDirectoryButtonClick(self):
        # After clicking the button, let the user choose a directory for saving
        self.directory_path = qt.QFileDialog.getExistingDirectory()

        self.New_Folder_Selected = True

        # Update the QT label with the directory path so the user can see it
        try:
            self.directoryButton.setText('...' + self.directory_path[-40:]) # Only show the last 40 characters for better display in the Slicer GUI
        except:
            self.directoryButton.setText(self.directory_path) # Show the entire path if it is less than 40 characters long

    def onDirectorySurfaceFittingButtonClick(self):
        # After clicking this button, let the user choose a directory
        # This folder should contain .PLY (or .STL) files of surfaces in correspondence with the bone displacment model surface
        # Use the training data 3D SLicer module to create these surfaces

        self.Directory_Input_Surfaces_Fitting = qt.QFileDialog.getExistingDirectory()

        # Update the QT label with the directory path so the user can see it
        try:
            self.directorySurfaceFittingButton.setText('...' + self.Directory_Input_Surfaces_Fitting[-40:]) # Only show the last 40 characters for better display in the Slicer GUI
        except:
            self.directorySurfaceFittingButton.setText(self.Directory_Input_Surfaces_Fitting) # Show the entire path if it is less than 40 characters long

    def onDirectoryFittingOutputButtonClick(self):
        # After clicking this button, let the user choose a directory to save the outfile text files to (from the fitting procedure)

        self.fitting_output_directory_path = qt.QFileDialog.getExistingDirectory()

        # Update the QT label with the directory path so the user can see it
        try:
            self.directoryFittingOutputButton.setText('...' + self.fitting_output_directory_path[-40:]) # Only show the last 40 characters for better display in the Slicer GUI
        except:
            self.directoryFittingOutputButton.setText(self.fitting_output_directory_path) # Show the entire path if it is less than 40 characters long


    def linspace(self, start, stop, num=50, endpoint=True, retstep=False):
        # Inspired from https://github.com/numpy/numpy/issues/2944

        import numpy.core.numeric as _nx

        # just pre-process start and stop a bit:
        start = np.asarray(start)[...,None]
        stop = np.asarray(stop)[...,None]
        num = int(num)
        if num <= 0:
            dtype = np.result_type(start, stop, 1.) # conserve dtype
            shape = np.broadcast(start, end).shape[:-1] + (0,)
            return np.empty(shape, dtype=dtype)
        if endpoint:
            if num == 1:
                dtype = np.result_type(start, stop, 1.)
                return start.astype(dtype)
            step = (stop-start)/float((num-1))
            y = _nx.arange(0, num) * step + start
            y[...,-1] = stop[...,0]
        else:
            step = (stop-start)/float(num)
            y = _nx.arange(0, num) * step + start
        if retstep:
            return y, step
        else:
            return y

    def Slicer_Landmark_To_List(self, Slicer_Landmarks):
        # Given a 3D Slicer markuplist, convert it to a Python list

        # Make a Python list of all the seed point locations
        numFids = Slicer_Landmarks.GetNumberOfFiducials()
        seedPoints = []
        # Create a list of the fiducial markers from the 'Markup List' input
        for i in range(numFids):
            ras = [0,0,0]
            Slicer_Landmarks.GetNthFiducialPosition(i,ras)
            seedPoints.append(ras)

        return seedPoints

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

    def onLoopCoefficientsButton(self):
        # Loop through each of the model coefficients one at a time. Should be very useful for visualization

        # Have the values increase and then decrease again (for a better visualization)
        values_decreasing_1 = np.linspace(0, -1*self.slider_range, self.loop_coefficients_num_steps);
        values_increasing = np.linspace(-1*self.slider_range, self.slider_range, self.loop_coefficients_num_steps);
        values_decreasing_2 = np.linspace(self.slider_range, 0, self.loop_coefficients_num_steps);

        # Concatinate the increasing and decreasing arrays together into a single vector
        model_cofficient_values = np.concatenate((values_decreasing_1, values_increasing, values_decreasing_2), axis=0)

        # Set the flag to hide the progress bar (for better visualization)
        self.show_progress_bar = False

        # Loop through the first 5 model coefficients
        for i in range(0, 5):

            # Set the reseting flag to true to stop onCompute() from running for each slider change
            self.reseting_state = True

            # Reset all the sliders back to zero
            self.FirstEVSlider.value  = 0
            self.SecondEVSlider.value = 0
            self.ThirdEVSlider.value  = 0
            self.FourthEVSlider.value = 0
            self.FifthEVSlider.value  = 0

            # Loop through the model coefficient values selected above
            for j in range(0, len(model_cofficient_values)):
                print(str(i) + str(model_cofficient_values[j]));

                # Change the value of the sliders now
                if   (i == 0):
                    self.FirstEVSlider.value  = model_cofficient_values[j]
                elif (i == 1):                    
                    self.SecondEVSlider.value = model_cofficient_values[j]
                elif (i == 2):                    
                    self.ThirdEVSlider.value  = model_cofficient_values[j]
                elif (i == 3):                    
                    self.FourthEVSlider.value = model_cofficient_values[j]
                elif (i == 4):                    
                    self.FifthEVSlider.value  = model_cofficient_values[j]

                # Set the reseting flag back to false now (if needed)
                self.reseting_state = False

                # Rebuild the model and create a new instance using the new EVs
                self.onCompute()

        # Flag to show the progress bar back to true
        self.show_progress_bar = True

    def onCompute(self):
        # This is the main function which updates the bone displacement model

        if self.directory_path == []:
            # Provide a popup error so the user knows that an input directory wasn't selected
            msg = qt.QMessageBox()
            msg.setIcon(qt.QMessageBox.Information)
            msg.setText('An input directory of training data was not provided.')
            msg.setInformativeText("Click on the Choose Training Data Folder to select the training data folder.")
            msg.setWindowTitle("Wrist PCA-Kinematics - Error")
            msg.show()

            raise ValueError('An input directory was not provided.')

            return;

        # Flag to check if OnCompute() is already running
        if self.Already_Running == True:
            # OnCompute() is already running so just return now
            return
        else:
            # If not already running (i.e. False) set the flag to true (i.e. is running)
            self.Already_Running = True

        # Show the status bar
        if (self.show_progress_bar == True):
            self.progressBar.show()
        else:
            self.progressBar.hide()

        slicer.app.processEvents()

        # If there is no PCA model create it
        # Or if a new folder is selected create the PCA model
        if self.pca_model == [] or self.New_Folder_Selected == True or self.New_Transform_Selected == True:

            # Reset the new folder selected flag back to false
            self.New_Folder_Selected = False

            # Reset the new transform selected flag back to false
            self.New_Transform_Selected = False

            # Load each STL file, process it, and save to a Python list (i.e. self.polydata_list)
            self.polydata_list, block_set = self.Load_Surface_From_Directory(self.directory_path, apply_tranform=True)

            # Update the status bar
            # Start at 50%
            self.progressBar.setValue(50) # Use 40% of the bar for this
            slicer.app.processEvents()
            slicer.util.showStatusMessage("Creating PCA Model...")

            # Update the number of eigenvalues to use to be the same as the number of poly_data minus one
            self.NumEVs = len(self.polydata_list) - 1

            # PCA filter
            self.pca_model = vtk.vtkPCAAnalysisFilter()
            self.pca_model.SetInputData(block_set)
            self.pca_model.Update()

            self.progressBar.setValue(75) 
            slicer.app.processEvents()
            slicer.util.showStatusMessage("Extracting Mean Position...")

            self.output_shape = vtk.vtkPolyData()

            try:
                self.output_shape.DeepCopy(self.polydata_list[0])
            except:
                print('Failed! Did you select a folder containing PLY files of surfaces in correspondence?')

            # Display the most important eigenvalues 
            print('Eigenvalues required for 100 percent explanation...')
            EVNo = self.pca_model.GetModesRequiredFor(1)
            eigenvalues_temp = []
            for j in range (0,EVNo):
                eigenvalues_temp.append(self.pca_model.GetEvals().GetValue(j))

            print(eigenvalues_temp)

        # Show the model at the specified scaling parameters
        params = vtk.vtkFloatArray()
        params.SetNumberOfComponents(1)
        params.SetNumberOfTuples(5) 
        params.SetTuple1(0,self.FirstEV)
        params.SetTuple1(1,self.SecondEV)
        params.SetTuple1(2,self.ThirdEV)
        params.SetTuple1(3,self.FourthEV)
        params.SetTuple1(4,self.FifthEV)

        self.progressBar.setValue(90) 
        slicer.app.processEvents()
        slicer.util.showStatusMessage("Applying Model Coefficients...")

        self.pca_model.GetParameterisedShape(params, self.output_shape)

        slicer.util.showStatusMessage("Model Coefficients Applied...")

        # Update the status bar
        # Start at 90%
        self.progressBar.setValue(90)
        slicer.app.processEvents()
        slicer.util.showStatusMessage("Outputting model...")

        # Create model node ("Model_Result") and add to scene
        self.Render_Surface(self.output_shape)

        # Set the status bar to 100%
        self.progressBar.setValue(100)

        # Hide the status bar
        self.progressBar.hide() 

        # Reset the status message on bottom of 3D Slicer
        slicer.util.showStatusMessage(" ")

        # Reset the already running flag back to false
        self.Already_Running = False

    def Render_Surface(self, polydata):
        # Push the given polydata to the 3D Slicer scene

        # Update the model (instead of creating a new one) if possible
        # By removing all the polydata beginning with PCA_
        # Only do this is the auto delete checkmark is checked

        if self.auto_delete.checked == True:
            # Delete the previous rendering of the surface
            node = slicer.util.getNode('Model_points')
            slicer.mrmlScene.RemoveNode(node)
            node = slicer.util.getNode('Model_wireframe')
            slicer.mrmlScene.RemoveNode(node)
            node = slicer.util.getNode('Model_surface')
            slicer.mrmlScene.RemoveNode(node)
            node = slicer.util.getNode('Normal_Vectors')
            slicer.mrmlScene.RemoveNode(node)

        if self.represent_points.checked == True:
            Model_Result = slicer.vtkMRMLModelNode()
            Model_Result.SetAndObservePolyData(polydata)
            Model_Result.SetName('Model_points')
            modelDisplay = slicer.vtkMRMLModelDisplayNode()

            # Represent the surface as a wireframe is the checkmark was selected by the user
            modelDisplay.SetRepresentation(0)

            modelDisplay.SetSliceIntersectionVisibility(True) # Show in slice view
            modelDisplay.SetVisibility(True) # Show in 3D view
            modelDisplay.SetName('Model_points')
            modelDisplay.BackfaceCullingOff()
            modelDisplay.SetColor(self.RedColor,self.GreenColor,self.BlueColor)
            slicer.mrmlScene.AddNode(modelDisplay)
            Model_Result.SetAndObserveDisplayNodeID(modelDisplay.GetID())
            slicer.mrmlScene.AddNode(Model_Result) 

        if self.represent_wireframe.checked == True:
            Model_Result = slicer.vtkMRMLModelNode()
            Model_Result.SetAndObservePolyData(polydata)
            Model_Result.SetName('Model_wireframe')
            modelDisplay = slicer.vtkMRMLModelDisplayNode()

            # Represent the surface as a wireframe is the checkmark was selected by the user
            modelDisplay.SetRepresentation(1)

            modelDisplay.SetSliceIntersectionVisibility(True) # Show in slice view
            modelDisplay.SetVisibility(True) # Show in 3D view
            modelDisplay.SetName('Model_wireframe')
            modelDisplay.BackfaceCullingOff()
            modelDisplay.SetColor(self.RedColor,self.GreenColor,self.BlueColor)
            slicer.mrmlScene.AddNode(modelDisplay)
            Model_Result.SetAndObserveDisplayNodeID(modelDisplay.GetID())
            slicer.mrmlScene.AddNode(Model_Result) 

        if self.represent_surface.checked == True:
            Model_Result = slicer.vtkMRMLModelNode()
            Model_Result.SetAndObservePolyData(polydata)
            Model_Result.SetName('Model_surface')
            modelDisplay = slicer.vtkMRMLModelDisplayNode()

            # Represent the surface as a wireframe is the checkmark was selected by the user
            modelDisplay.SetRepresentation(2)

            modelDisplay.SetSliceIntersectionVisibility(True) # Show in slice view
            modelDisplay.SetVisibility(True) # Show in 3D view
            modelDisplay.SetName('Model_surface')
            modelDisplay.BackfaceCullingOff()
            modelDisplay.SetColor(self.RedColor,self.GreenColor,self.BlueColor)
            slicer.mrmlScene.AddNode(modelDisplay)
            Model_Result.SetAndObserveDisplayNodeID(modelDisplay.GetID())
            slicer.mrmlScene.AddNode(Model_Result) 

        if self.represent_surface_edges.checked == True:
            Model_Result = slicer.vtkMRMLModelNode()
            Model_Result.SetAndObservePolyData(polydata)
            Model_Result.SetName('Model_surface')
            modelDisplay = slicer.vtkMRMLModelDisplayNode()
            modelDisplay.EdgeVisibilityOn()

            # Represent the surface as a wireframe is the checkmark was selected by the user
            modelDisplay.SetRepresentation(2)

            modelDisplay.SetSliceIntersectionVisibility(True) # Show in slice view
            modelDisplay.SetVisibility(True) # Show in 3D view
            modelDisplay.SetName('Model_surface')
            modelDisplay.BackfaceCullingOff()
            modelDisplay.SetColor(self.RedColor,self.GreenColor,self.BlueColor)
            slicer.mrmlScene.AddNode(modelDisplay)
            Model_Result.SetAndObserveDisplayNodeID(modelDisplay.GetID())
            slicer.mrmlScene.AddNode(Model_Result) 

        # Show a vector (i.e. glyph) of the normal at each point along the surface
        if self.show_glyph.checked == True:
            glyph = self.CreateGlyphs(polydata, True)

            scalarRangeElevation = polydata.GetScalarRange()

            Model_Result = slicer.vtkMRMLModelNode()
            Model_Result.SetAndObservePolyData(glyph.GetOutput())
            Model_Result.SetName('Normal_Vectors')

            modelDisplay = slicer.vtkMRMLModelDisplayNode()

            if self.represent_wireframe.checked == True:
                # Represent the surface as a wireframe is the checkmark was selected by the user
                modelDisplay.SetRepresentation(1)

            modelDisplay.SetScalarRange(scalarRangeElevation)
            modelDisplay.SetScalarVisibility(True)
            modelDisplay.SetSliceIntersectionVisibility(True) # Show in slice view
            modelDisplay.SetVisibility(True) # Show in 3D view
            modelDisplay.SetName('Normal_Vectors')
            modelDisplay.BackfaceCullingOff()
            modelDisplay.SetActiveScalarName('VectorMagnitude')
            slicer.mrmlScene.AddNode(modelDisplay)
            Model_Result.SetAndObserveDisplayNodeID(modelDisplay.GetID())
            slicer.mrmlScene.AddNode(Model_Result) 

            return

    def Fit_Polydata(self, pca_model, polydata_list, num_tuples):
        # Use the PCA model to find the best fitting scaling components
        # for each polydata in the polydata_list
        # Output the coefficients to a text file

        # Import the linear algebra package from numpy (for distance measurement)
        from numpy import linalg as LA

        # Initilize the vtkFloatArray to hold the fitted parameters
        fitted_params = vtk.vtkFloatArray()
        fitted_params.SetNumberOfComponents(1)
        fitted_params.SetNumberOfTuples(num_tuples)

        # Initilize a Python list for the coefficients
        coefficients = np.zeros(num_tuples)

        for i in range(0, len(polydata_list)):
            print(' ')
            print('Fitting polydata # ' + str(i))

            # Find the best fitting scaling paramters
            pca_model.GetShapeParameters(polydata_list[i], fitted_params, num_tuples)

            print('fitted_params for ' + self.files[i] + ': ')

            new_coeff = vtk_to_numpy(fitted_params)
            print(new_coeff)

            # Add the new coefficients to the array of all the coefficients
            coefficients = np.vstack((coefficients,new_coeff))

        # Save the coefficients for interpolating between them later on
        self.fitted_coefficients = coefficients

        # The first row is all zeros so just delete is
        self.fitted_coefficients = np.delete(self.fitted_coefficients, [0],axis=0)           

        # Set the status bar to 100%
        self.progressBar.setValue(0)

        # Show the status bar
        self.progressBar.show() 

       
        # Remove the first row of zeros (used to initilize the array)
        coefficients = np.delete(coefficients, (0), axis=0)

        # Output the coefficients to a text file using numpy
        np.savetxt(os.path.join(self.fitting_output_directory_path, 'Fitted_Model_Coefficients.txt'), coefficients, delimiter=',',  fmt='%f')

        # We need to know which surface file corresponds to which coefficent
        # This text file will list the order the surfaces in the coefficient text file
        path = os.path.join(self.fitting_output_directory_path,'Files_Fitted.txt')
        with open(path, 'w') as file_handler:
            for item in self.files:
                file_handler.write("{}\n".format(item))

        # Hide the status bar
        self.progressBar.hide() 

        # Update status bar message
        slicer.util.showStatusMessage('Saved fitted model coefficients to a text file')

        slicer.app.processEvents()

    def Load_Surface_From_Directory(self, directory_path, apply_tranform=True):
        # Given some directory file path, load all the suface files (either PLY or STL)
        # Save the surfaces to a Python list

        # Find all the files in the input folder
        files = os.listdir(directory_path)
        files.sort(key=self.alphanum_key)

        # How many eigenvalues to use when fitting
        self.num_tuples = self.NumEVs #len(self.files)

        # Create a multiblock dataset to hold all of the surface polydata
        block_set = vtk.vtkMultiBlockDataSet()

        # Initilize Python list to hold all of the polydata
        polydata_list = []

        # List to keep track of all the useable files in the folder
        self.files = []

        # Variable to get the ploydata vtkMultiBlockDataSet order correct
        iter = 0

        # Load each .stl file and add to the block set of the polydata
        for i in range(0, len(files)):

            # Update the status bar
            self.progressBar.setValue(float(i)/len(files)*100) # Use 100% of the bar for this
            slicer.app.processEvents()
            slicer.util.showStatusMessage("Loading STL Surfaces...")

            filename = os.path.join(directory_path, files[i])         

            if files[i][-3:] == 'stl':
                # Save the file name for later on 
                self.files.append(files[i])

                reader = vtk.vtkSTLReader()
                reader.SetFileName(filename)
                reader.Update()
                poly_data = reader.GetOutput()

                # Save the loaded poly_data to the list
                polydata_list.append(poly_data)

            elif files[i][-3:] == 'ply':
                # Save the file name for later on 
                self.files.append(files[i])

                reader = vtk.vtkPLYReader()
                reader.SetFileName(filename)
                reader.Update()
                poly_data = reader.GetOutput()

                # Save the loaded poly_data to the list
                polydata_list.append(poly_data)

            else:
                # Skip this iteration since the file is not a STL or a PLY file
                continue
                print('File type provided was ' + files[i][-3:] + ' but only PLY and STL files are currently support. Skipping this file.')  

            block_set.SetBlock(iter, poly_data)
            
            # Add one to the iteration variable
            iter = iter + 1

        # Check to see if all the polydata have the same number of points
        # If they are not exactly the same, Slicer will just crash with no error message
        # which is difficult to debug
        polydata_points = []
        for i in range(0,len(polydata_list)):
            points = polydata_list[i].GetPoints()
            polydata_points.append(points.GetNumberOfPoints())
        
        # Now check to see if all the number of points are the same
        # If it is not the same then the number of unique point numbers will not be 
        # equal to one
        if len(np.unique(np.asarray(polydata_points))) != 1:
            # If the lengths are not the same raise an error and provide a suggested solution
            msg = qt.QMessageBox()
            msg.setIcon(qt.QMessageBox.Information)
            msg.setText("The number of surface points (i.e. verticies) is not the same for all of the training data!")
            msg.setInformativeText("Click on Show Details for a suggested fix.")
            msg.setWindowTitle("Create PCA Model - Error")
            msg.setDetailedText("The training data will need to be recreated or else remove the surfaces which have a different number of points."+
                            " \n \n You can check the number of points by loading the surfaces into Slicer and using the builtin Model module.")
            msg.show()

            raise ValueError("The number of surface points (i.e. verticies) is not the same for all of the training data!")

        return polydata_list, block_set

    def onPCAFittingButtonClicked(self):
        # Load the surfaces in the folder of surfaces in correspondence
        # Fit the PCA kinematic model to each surface by projecting the points onto the eigenvectors
        # Save the eigenvalue coefficients to a text file for processing in other programs (such as Matlab)

        # Check if we have the required data
        if self.Directory_Input_Surfaces_Fitting == []:

            # Provide a popup error so the user knows that an input directory wasn't selected
            msg = qt.QMessageBox()
            msg.setIcon(qt.QMessageBox.Information)
            msg.setText('A directory of surfaces for fitting was not provided.')
            msg.setInformativeText("Click on the Choose Input Folder to select the folder of surfaces for fitting.")
            msg.setWindowTitle("Wrist PCA-Kinematics - Error")
            msg.show()

            raise ValueError('A directory of surfaces for fitting was not provided.')

            return;

        if self.pca_model == []:

            # Provide a popup error so the user knows that a bone displacement model has not been created yet
            msg = qt.QMessageBox()
            msg.setIcon(qt.QMessageBox.Information)
            msg.setText('A bone displacement model has not been created yet.')
            msg.setInformativeText("Click on the Create Bone Displacement Model button first to create the model.")
            msg.setWindowTitle("Wrist PCA-Kinematics - Error")
            msg.show()

            raise ValueError('A bone displacement model has not been created yet.')

            return;
        
        if self.fitting_output_directory_path == []:

            # Provide a popup error so the user knows that an output directory wasn't selected
            msg = qt.QMessageBox()
            msg.setIcon(qt.QMessageBox.Information)
            msg.setText('An output directory to save the text files to was not provided.')
            msg.setInformativeText("Click on the Choose Output Folder to select the folder to save to.")
            msg.setWindowTitle("Wrist PCA-Kinematics - Error")
            msg.show()

            raise ValueError('An output directory to save the text files to was not provided.')

            return;



        # Show the status bar
        self.progressBar.show()
        slicer.app.processEvents()
        
        # Load each STL file, process it, and save to a Python list (i.e. self.polydata_list)
        # The folder of the STL files to fit the model to is "self.Directory_Input_Surfaces_Fitting"
        # apply_tranform = False for now, but this might need to be True for later?
        self.polydata_list, block_set = self.Load_Surface_From_Directory(self.Directory_Input_Surfaces_Fitting, apply_tranform=False)

        # Fit the model to the polydata find the scaling parameters
        self.Fit_Polydata(self.pca_model, self.polydata_list, self.num_tuples)

        shape = self.fitted_coefficients.shape


    def onTimeSelectSlider_FittedChange(self):
        # After fitting the model to the surfaces in the Run Fitting Procedure folder, 
        # interpolate between the found coefficients

        # Has a bone displacement model been created already?
        if self.pca_model == []:

            # Provide a popup error so the user knows that a bone displacement model has not been created yet
            msg = qt.QMessageBox()
            msg.setIcon(qt.QMessageBox.Information)
            msg.setText('A bone displacement model has not been created yet.')
            msg.setInformativeText("Click on the Create Bone Displacement Model button first to create the model.")
            msg.setWindowTitle("Wrist PCA-Kinematics - Error")
            msg.show()

            raise ValueError('A bone displacement model has not been created yet.')

            return;

        
        # Flag to check if onTimeSelectSlider_FittedChange() is already running
        if self.Already_Running == True:
            # onTimeSelectSlider_FittedChange() is already running so just return now
            return
        else:
            # If not already running (i.e. False) set the flag to true (i.e. is running)
            self.Already_Running = True

        # Save the current value of the slider
        self.TimeSelected_Fitted = self.TimeSelectSlider_Fitted.value

        shape = self.fitted_coefficients.shape

        # Fit a curve to each of eigenvalue list seperately
        fitted_EV = []
        for i in range(0,shape[1]):

            # Equally space the x values to go from -1 to 1 with the number of fitted positions
            x = np.linspace(-1,1,shape[0])
            y = self.fitted_coefficients[:,i]
            z = np.polyfit(x,y,self.FittingOrder_Fitted) # self.FittingOrder_Fitted is 1 for linear, 2 for parabolic, etc.
            p_fit = np.poly1d(z)

            # Apply the model at the time point selected using the slider
            fitted_EV.append(p_fit(self.TimeSelected_Fitted))

        print('fitted_EV')
        print(fitted_EV)

        # Use the fitted eigenvalues to update the sliders and the model                
        # Don't do the autorun if the sliders are resetting
        self.reseting_state = True

        # Reset all the sliders back to zero
        self.FirstEVSlider.value  = fitted_EV[0]
        self.SecondEVSlider.value = fitted_EV[1]
        self.ThirdEVSlider.value  = fitted_EV[2]
        self.FourthEVSlider.value = fitted_EV[3]
        self.FifthEVSlider.value  = fitted_EV[4]

        # Set the reseting flag back to false now (if needed)
        self.reseting_state = False

        # Show the model at the specified scaling parameters
        params = vtk.vtkFloatArray()
        params.SetNumberOfComponents(1)
        params.SetNumberOfTuples(len(fitted_EV)) 

        for i in range(0,len(fitted_EV)):
            params.SetTuple1(i,fitted_EV[i])

        self.progressBar.setValue(50) 
        slicer.app.processEvents()
        slicer.util.showStatusMessage("Applying Model Coefficients...")

        self.pca_model.GetParameterisedShape(params, self.output_shape)

        # Update the status bar
        # Start at 90%
        self.progressBar.setValue(90)
        slicer.app.processEvents()
        slicer.util.showStatusMessage("Outputting model...")

        # Create model node ("Model_Result") and add to scene
        self.Render_Surface(self.output_shape)

        # Reset the already running flag back to false
        self.Already_Running = False

        return

    def onFittingOrderSlider_FittedChange(self, newValue):
        # Save the current state of the slider
        self.FittingOrder_Fitted = newValue

if __name__ == "__main__":
    # TODO: need a way to access and parse command line arguments
    # TODO: ideally command line args should handle --xml

    import sys
    print(sys.argv)

