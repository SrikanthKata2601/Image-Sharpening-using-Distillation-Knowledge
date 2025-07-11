# Image-Sharpening-using-Knowledge-Distillation

This project implements a deep learning model to *sharpen blurred images* using a *lightweight U-Net architecture* enhanced with *edge detection and segmentation maps. The model is designed for **efficient deployment* on low-resource devices and supports visual quality evaluation using *SSIM* and *PSNR* metrics.

---

##  Project Overview

- Takes blurred images as input  
- Extracts edge maps (Sobel filters) and segmentation masks (HSV thresholding)  
- Forms a 5-channel input: RGB + Edge + Segmentation  
- Processes through a compact U-Net to output a sharpened version of the image  
- Evaluated using *SSIM, **PSNR*, and qualitative side-by-side comparisons

---

##  Technologies Used

- Python
- PyTorch
- OpenCV
- torchvision
- matplotlib
- pytorch-msssim
- Google Colab (for development & training)

---

##  Installation

Install the dependencies:

bash
pip install torch torchvision opencv-python matplotlib pytorch-msssim


---

##  Dataset

- *DIV2K (DIVerse 2K resolution images)* is used as the source for high-resolution sharp images.
- During training, synthetic Gaussian blur is applied to generate input-output pairs.

🔗 Dataset link: [DIV2K on Kaggle](https://www.kaggle.com/datasets/joe1995/div2k-dataset)

---

##  Training

To train the model on your dataset of *sharp images*:

bash
python train_unet_edge_seg.py


Key Features:
- Applies Gaussian blur to sharp images
- Extracts edge + segmentation maps for guidance
- Trains a 5-channel U-Net with hybrid MSE + SSIM loss
- Visualizes and saves output images (original | blurred | deblurred)

---

##  Testing

To deblur a folder of *already blurred* test images:

bash
python test_unet_edge_seg.py


- Accepts blurred images only (no need for ground truth)
- Automatically generates edge and segmentation maps
- Produces visual output showing deblurred images side-by-side

---

##  Evaluation Metrics

- *SSIM* (Structural Similarity Index)
- *PSNR* (Peak Signal-to-Noise Ratio)
- Loss Function:  
  0.4 * MSE + 0.6 * (1 - SSIM)

---

##  Future Work

- Export student model to *ONNX*
- Optimize using *OpenVINO* for deployment on CPUs and edge devices
- Validate performance on *Raspberry Pi* or low-power laptops

---

##  Team Members

- *S. Sai Tharuneswar*
- *K. Srikanth*

---

##  References

- [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)
- [pytorch-msssim GitHub](https://github.com/VainF/pytorch-msssim)
- OpenCV and torchvision documentation

---

##  License

This project is for academic and research purposes only.
