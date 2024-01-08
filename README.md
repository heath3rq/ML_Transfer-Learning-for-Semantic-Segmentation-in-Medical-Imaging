# ECE661GroupProject_TransferLearning

**1. Prerequisites**

*Note: The following instructions are provided for environment setup in GitHub Codespaces. For other environments, refer to the official documentation for [MMSegmentation](https://mmsegmentation.readthedocs.io/en/latest/get_started.html).*

   + **Step 1:** Create a Conda environment and activate it in the terminal.

        ```
        conda create --name openmmlab python=3.10 -y
        conda activate openmmlab
        ```

        *Note: If encountering issues with `conda activate`, try running `conda init bash` in the terminal and then restart the terminal.*

   + **Step 2:** Run all cells in the [1_setup.ipynb](1_setup.ipynb).


**2. Data Preprocess**

   + **Step 1:** Download [DME data](https://www.kaggle.com/code/atrichatterjee7/unet-imagesegmentation/input) from Kaggle

   + **Step 2:** Run [2_data_preprocess.ipynb](2_data_preprocess.ipynb) to convert `.mat` format data to `.png`. If you need the image to be a specific size, complete the TO-DO item in the notebook.  

   + **Step 3:** Run [3_split_data.py](3_split_data.py) to split data into train and test sets. 


**3. Inference & Fine-tuning**

   + **Step 1:** Run [4_inference.ipynb](4_inference.ipynb) to make inference using different pre-trained models. 

   + **Step 2:** Run [5_modeling.ipynb](5_modeling.ipynb) to re-train the six top-performing pre-trained models identified in the previous step on the DME dataset. 

   + **Step 3:** Run [6_finetuning.ipynb](6_finetuning.ipynb) to fine-tune the top-performing model identified in the previous step.

**4. Results & Conclusion**

In our analysis, we extensively fine-tuned the UNet model to enhance performance, maximizing mDice, mIoU, and mFscore. Fine-tuning iterations involved variations in crop size (256x256, 288x288, 320x320), model architecture (UNet with DeepLabv3, FCN, or PSPNet heads), and loss functions (cross-entropy, dice, and focal loss).

These changes, tailored to pre-trained UNet models, addressed compatibility concerns with the limited dataset (110 OCT images). Computational challenges arose when augmenting crop size or iterations. Additional experiments with diverse learning rates, the Adam optimizer, and freezing early layers did not yield significant improvements and are excluded from this report. The conclusive fine-tuned results, based on a custom UNet configuration, are presented in the table below:

# Final Fine-Tuned Results on Custom Configurations

| Model | mDice | mAccuracy | mIoU | mFscore |
| ------ | ------ | --------- | ---- | ------- |
| UNet with FCN [crop size: 256*256, 200 iters, cross entropy loss] | 73.82 | 68.77 | 65.44 | 73.82 |
| UNet with DeepLabV3 [crop size: 256*256, 200 iters, cross entropy loss] | 73.63 | 67.53 | 65.28 | 73.63 |
| UNet with FCN [crop size: 320*320, 200 iters, cross entropy loss] | 77.44 | 75.37 | 68.72 | 77.44 |
| UNet with FCN [crop size: 288*288, 200 iters, cross entropy loss] | 77.37 | 78.26 | 68.32 | 77.37 |
| UNet with FCN [crop size: 256*256, 150 iters, cross entropy loss] | 71.38 | 65.40 | 63.39 | 71.38 |
| UNet with FCN [crop size: 256*256, 300 iters, cross entropy loss] | 65.70 | 60.04 | 59.07 | 65.70 |
| UNet with FCN [crop size: 320*320, 200 iters, focal loss] | 77.40 | 76.22 | 68.70 | 77.44 |
| UNet with FCN [crop size: 320*320, 200 iters, dice loss] | 77.63 | 78.74 | 68.90 | 77.63 |
| UNet with DeepLabV3 [crop size: 320*320, 200 iters, dice loss] | 79.38 | 77.44 | 70.64 | 79.38 |

* m[MetricName] represents the average of all classes

The final model included a UNet backbone with DeepLabV3 heads, images cropped to (320x320), and trained over 200 iterations with dice loss. UNet was originally introduced for medical semantic segmentation and is trained on thousands of labeled data samples, specifically retinal images. It implements a decoder-encoder network that can extract general information features the deeper in the network it traverses. It also included skip connections which reintroduced detailed features back into the decoder rather than forgetting that information throughout the layers to segment properly. From the visualized examples in Section \ref{inference}, very few pixels are fluid so as much information must be retained from each layer. Although the other models we tried had higher performance during inference, they were likely predicting all pixels as backgrounds rather than distinguishing the fluid. With initial fine-tuning with all the models, we uncovered that UNet was able to transfer knowledge from its pre-trained weights effectively as it was trained on medical images, and appropriately differentiated fluid from the background. 
