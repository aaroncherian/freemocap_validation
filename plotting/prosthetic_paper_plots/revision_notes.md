### Main revisions made so far

- **Reframed the paper around the FMC-Hybrid approach and standardized the terminology throughout.**
  - The paper now consistently distinguishes **FMC-MediaPipe**, **FMC-RTMPose**, **FMC-Hybrid** (RTMPose whole-body tracking + prosthesis-specific DLC landmarks), and **Qualisys**.
  - This makes it much clearer what is actually being compared and what the prosthesis-specific model contributes.
  - **Addresses:** reviewer concerns about unclear system/model terminology and the need to separate performance of the pretrained models from the prosthesis-specific approach.

- **Substantially expanded the DeepLabCut / prosthesis-specific model methods.**
  - Added details on the training dataset, including the number of labeled frames, how they were selected, and how they were split into training and test sets.
  - Explicitly stated that the trained model was applied without additional training to all alignment conditions, including conditions not represented in the labeled dataset.
  - **Addresses:** reviewer concerns about reproducibility, insufficient model-training detail, and whether the model was simply trained on every condition it was later evaluated on.

- **Expanded the comparisons to include the pretrained pose-estimation approaches, not just FMC-Hybrid.**
  - Regenerated the major results figures with consistent FMC-Hybrid terminology.
  - Generated additional versions showing FMC-MediaPipe and FMC-RTMPose alongside FMC-Hybrid and Qualisys (in supplementary material).
  - **Addresses:** reviewer requests to show how the general-purpose models performed and to demonstrate what the prosthesis-specific tracking adds.

- **Clarified and strengthened the biomechanical outcome definitions and corresponding figures.**
  - Prosthetic shank length is now explicitly defined as the 3D knee-to-ankle distance.
  - Joint-angle coordinate systems are described in more detail.


- **Added a more nuanced discussion of the clinical feasibility of this particular pipeline.**
  - The revision now adddresses the challenges associated with implementing this pipeline in a clinical setting, considering the barriers posed by manual labeling, custom model training, computational resources, and pose-estimation expertise.
  - Makes a note on how we still need pose estimation software that can track residual limbs better
  - **Addresses:** reviewer concerns that the original discussion overstated near-term clinical feasibility.

- **Strengthened the limitations and generalizability discussion.**
  - Added explicit discussion of the single-participant design.
  - Expanded DLC limitations related to prosthesis-specific training and generalizability to other prosthetic designs, amputation levels, and gait patterns, and how these limitations could affect widespread clinical adoption.
  - **Addresses:** reviewer concerns about generalizability and the limits of drawing broad clinical conclusions from a single-subject proof-of-concept study.


- **Revised the overall framing toward a proof-of-concept evaluation rather than a broad validation claim.**
  - Changed the title to better reflect the study design and level of evidence, though not sure if its the final one 
  - The Methods and Discussion now explicitly describe the work as a single-participant proof-of-concept.
  - We are also considering changing the title to:
  - **Addresses:** reviewer concern that the scope/title should better reflect the study design and level of evidence.

