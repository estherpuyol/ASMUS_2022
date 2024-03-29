# AI-enabled Assessment of Cardiac Systolic and Diastolic Function from Echocardiography

Left ventricular (LV) function is an important factor in terms of patient management, outcome, and long-term survival of patients with heart disease. The most recently published clinical guidelines for heart failure recognise that over reliance on only one measure of cardiac function (LV ejection fraction) as a diagnostic and treatment stratification biomarker is suboptimal. Recent advances in AI-based echocardiography analysis have shown excellent results on automated estimation of LV volumes and LV ejection fraction. However, from time-varying 2-D echocardiography acquisition, a richer description of cardiac function can be obtained by estimating functional biomarkers from the complete cardiac cycle. In this work we propose for the first time an AI approach for deriving advanced biomarkers of systolic and diastolic LV function from 2-D echocardiography based on segmentations of the full cardiac cycle. These biomarkers will allow clinicians to obtain a much richer picture of the heart in health and disease. The AI model is based on the 'nn-Unet' framework and was trained and tested using four different databases. Results show excellent agreement between manual and automated analysis and showcase the potential of the advanced systolic and diastolic biomarkers for patient stratification. Finally, for a subset of 50 cases, we perform a correlation analysis between clinical biomarkers derived from echocardiography and CMR and we show excellent agreement between the two modalities. 

## Code
This repository only provides the code and the model weights for the segmentation network. Since the GSTFT data set cannot be made publicly available due to restricted access under hospital ethics and because informed consent from participants did not cover public deposition of data, we shared only the model trained with EchoNet-Dynamic and Cardiac Acquisitions for Multi-structure Ultrasound Segmentation (CAMUS).

## Citation
If our work brings insights to you, or you use the codebase, please cite our paper as:
```
@artical{puyol2022ASMUS,
  title={AI-enabled Assessment of Cardiac Systolic and Diastolic Function from Echocardiography},
  author={Puyol-Antó Esther, Ruijsink Bram, Sidhu Baldeep S., Gould Justin, Porter Bradley, Elliott Mark K., Mehta Vishal, Gu Haotian, Rinaldi1,2 Martin Cowie5 Phil Chowienczyk Christopher A., Razavi Reza, and King Andrew P.},
  booktitle={International Workshop on Advances in Simplifying Medical Ultrasound},
  year={2022}
}
```

## Acknowledgement
The backbone, training and test scripts are mainly based on the nnU-Net framework: https://github.com/MIC-DKFZ/nnUNet
