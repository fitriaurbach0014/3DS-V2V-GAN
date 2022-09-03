# Pseudo Slicer on Three Dimensional Brain Tumor Segmentation

A constructive way to assist surgeons before performing brain tumor surgery is by visualizing a three-dimensional MRI image. Specifically, we anticipate the volume
of a brain tumor during a pre-operative (pre-op) examination. However, the availability of 3D MRI brain tumors is very limited, especially on our case. Hence, we attempt to propose a model to generate a pseudo slicer to complete 2D image slices into 3D ones by integrating image slice interpolation and 3D volume-to-volume
generative adversarial networks (3D-V2V-GAN). Therefore, we call the proposed model 3DS-V2V-GAN. In short, we interpolate each slice 2D-MRI and then segment all slices to obtain the volume of the brain tumor. In this work, we evaluated the proposed model on 2D MRI meningioma of 27 patients from a university hospital, Indonesia. In the evaluations, the proposed model showed the generated brain tumor segmentation similar to the ground truth. Additionally, we satisfy to have radiologists
affirming good quality to the segmentation results.

Reference:
[Vox2Vox: 3D-GAN for Brain Tumor Segmentation] (https://arxiv.org/abs/2003.13653)
