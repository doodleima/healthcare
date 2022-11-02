# healthcare

SourceCode files which related to the medical AI area.

# mri_preprocess

  I. Dicom file converter : dicom to image (use DiCom ToolKit & pydicom - pydicom package only)
     1) Restful API: performed to send & convert images inside the docker container
  II. Data Preprocess(Total): Apply Threshold, CLAHE, Resample(Reshape)...


# pytorch_train

  I. Brain Segmentation using MONAI applied-Model : Brain MR Images with multi classes
     1) monai: monai_based
     2) torchio: apply torchio into DataLoader