



# forest_semantic_segmentation



## Dinis S. Simões, Afonso E. Carvalho and David Portugal



This project focuses on developing a machine learning-based semantic segmentation model specifically tuned for forest environments. The goal is to segment the scene into relevant classes to support robotic operations in such settings.



Our main contribution was to modify the classification layer of the [shi-labs/oneformer_ade20k_dinat_large](https://huggingface.co/shi-labs/oneformer_ade20k_dinat_large) method to define a new set of 13 semantic classes, replacing the original 150.



The new set of custom semantic classes that was considered is:

- other;
- soil;
- trunk:
- water;
- vegetation;
- low grass;
- high grass;
- stone;
- stump;
- person;
- animal;
- canopy;
- mud.



We developed two variants of the proposed model:

- ***forest_dinat_g***: trained exclusively on the [GOOSE](https://goose-dataset.de/) real-world forest dataset;

- ***forest_dinat_g_cwt***: trained on a combination of the [GOOSE](https://goose-dataset.de/) real-world forest dataset and the [CWT](https://gamma.umd.edu/tns/) real-world non-forest dataset.





The provided codes were developed in Python and correspond to the following:



- ***architecture_modification***: Adjusts the classification layer of the original method to define a new set of semantic classes;



- ***training***: Trains the modified architecture;



- ***inference***: Performs inference on a set of test images using the trained model.



The project was implemented and tested on Ubuntu 20.04.





### **Installation**





- Make sure the following dependencies are installed (although different versions may work, these are the ones that were tested):



    1. pytorch==2.1.2  

    2. cuda==12.1  

    3. pip install transformers  

    4. pip install natten==0.14.6  

    5. pip install scipy  

    6. pip install optuna  

    7. pip install wandb  

    8. pip install scikit-learn 



- Sign up for and log in to [Weights & Biases](https://wandb.ai) (W&B) -- optional.



### **Notes**



- If you are using Docker, please make sure all requirements are met to be able to use the NVIDIA GPU inside the container, and refer to [this](https://hub.docker.com/layers/pytorch/pytorch/2.1.2-cuda12.1-cudnn8-devel/images/sha256:a5de097b482f5927baf2322f4419f11044bfe4f08c7b7593dbaff8e41d03a204) image;



- W&B is only used to monitor the training progress. It is not essential for running the code and can be removed if desired;



- In the ***.yaml*** files, it is required to provide information about the directory paths to:


    1. Save the modified architecture;

    2. Access training RGB images;

    3. Access training grayscale masks;

    4. Access validation RGB images;

    5. Access validation grayscale masks;

    6. Save the trained model;

    7. Access test RGB images;

    8. Save the segmented RGB images;




- In every script, it is necessary to specify the directory path to access the corresponding ***.yaml*** file;


- Users can configure:



    1. The number of semantic classes (defined in the ***architecture_modification.yaml*** file);

    2. The range of hyper-parameters values (defined in the ***training.yaml*** file);

    3. The early stopping mechanism (defined in the ***training.yaml*** file);

    4. The number of training trials (defined in the ***training.yaml*** file);

    5. What information is sent to W&B (directly defined in the ***training.py*** script).



### **To Do**

- Integrate ROS2 wrapper;

- Make the *forest_dinat* trained model publicly available;

- Retrain the model, focusing on classes such as **stump**, **person**, and **animal**, which were underrepresented in the original training dataset and may cause difficulties in identifying these elements.



### **Project Video**

[![YouTube](http://i.ytimg.com/vi/IFs4Nn2fgKE/hqdefault.jpg)](https://www.youtube.com/watch?v=IFs4Nn2fgKE)



### **Access to Models**

The modified architecture, as well as the ***forest_dinat_g*** and ***forest_dinat_g_cwt*** models, can be downloaded here:

https://drive.google.com/drive/folders/1YSDCQNTqj44fIiFqyNlAtjZ15Ewtp75q?usp=drive_link