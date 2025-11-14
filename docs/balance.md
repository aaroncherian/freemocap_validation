## Overview
Patient balance and progress in rehabilitation is often measured using comprehensive  **qualitative** assessments. However, these tests 
tend to be **subjective** and may not be responsive to small changes in a patient. The ability to capture human movement data while performing these clinical assessments could provide clinicians with quantitative, standardized insight into patient movement capability. 

To validate the suitability of FreeMoCap as a clinical assessment tool, we record participants performing the NIH Standing Balance Test (SBT). We then compare center of mass (COM) derived parameters based on 3D pose estimation from FreeMoCap to 3D estimates from Qualisys, a marker-based system.

### The NIH Standing Balance Test (SBT)

The NIH SBT is an assessment tool designed to evaluate an individual’s postural stability and balance. Participants stand feet together, and are asked to stand as still as possible for 50s under increasingly difficult conditions. These conditions include:

1) Standing with eyes **open** on **solid ground** <br>
2) Standing with eyes **closed** on **solid ground** <br>
3) Standing with eyes **open** on a **foam pad** <br>
4) Standing with eyes **closed** on a **foam pad** <br>

In a typical assessment, an accelerometer (usually an iPhone) is worn around the participant's waist. The accelerometer measures postural sway, which is then converted into a number of NIH scores that represent overall balance ability 

## Methods

### Task: The NIH Standing Balance Test (SBT) 
Participants were asked to complete three trials of the NIH SBT. For each condition, participants were instructed to stand as still as possible for 55 seconds, keeping their gaze fixated on a specific point, feet together and arms held at their side. Participants were recorded using three different systems, detailed below. 

### Tracking Patient Motion

#### Qualisys
Retroreflective markers were placed on the participant, and a Qualisys marker-based system was used to capture motion capture data

#### FreeMoCap
Six webcams were set up around the subject. Cameras were calibrated, and then used to record the patient during the SBT. 

### Data Analysis

#### Reconstructing 3D Data
Synchronized videos from the webcams were fed through the FreeMoCap software to reconstruct 3D joint centers. FreeMoCap data was smoothed using a low-pass, 4th order, 6Hz Butterworth filter. Qualisys data was downsampled and time-synchronized with FreeMoCap data. Specific frames were annotated for the start and end point of each balance condition within the recording. 1600 frames were analyzed for each condition. 

#### Center of Mass Calculation
For both systems, segment and total body center of mass was calculated using anthropometric data. For each condition, the overall path length of the center of mass was calculated. Center of mass position was also used to calculate center of mass velocity as well during each condition. 

## Results
### Center of mass dispersion

The 2d ground plane projection of the center of mass shows increased dispersion during progressively harder stances.

<iframe width="650" height="650" frameborder="0" scrolling="no" src="../balance_data/com_dispersion_plots.html" ></iframe>

### Normalized path length

Center of mass path length shows a generally increasing trend across stances.

<iframe width="700" height="650" frameborder="0" scrolling="no" src='../balance_data/path_length_line_plots.html'> </iframe>

### COM Position and Velocity
<iframe width="1000" height="800" frameborder="0" scrolling="no" src='../balance_data/com_position_and_velocity.html'> </iframe>

### COM Velocity Distribution
You can explore the distribution of center of mass velocity for each stance condition below.

<iframe
  src = "../balance_data/com_velocity_violin_x.html",
  style="width:1000px; height:600px; border:none;"
  loading="lazy", 
  scrolling = "no"
  ></iframe>

<iframe
  src = "../balance_data/com_velocity_violin_y.html",
  style="width:1000px; height:600px; border:none;"
  loading="lazy"
  scrolling = "no"
  ></iframe>

<iframe
  src = "../balance_data/com_velocity_violin_z.html",
  style="width:1000px; height:600px; border:none;"
  loading="lazy"
  scrolling = "no"
  ></iframe>