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


**3. Training**

   + **Step 1:** Run [4_inference.ipynb](4_inference.ipynb) to make inference using different pre-trained models. 

   + **Step 2:** Run [5_modeling.ipynb](5_modeling.ipynb) to re-train the six top-performing pre-trained models identified in the previous step on the DME dataset. 

   + **Step 3:** Run [6_finetuning.ipynb](6_finetuning.ipynb) to fine-tune the top-performing model identified in the previous step. 