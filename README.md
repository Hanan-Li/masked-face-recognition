# Masked Face Verification
## Project Directory:
Note, some folders mightve been gitignored

* **image_comparison/**  
   Contains statistics and examples for facial verification for pairwise evaluation for each pair in mixed_face_dataset_subset
* **lfw_complete/**  
    Complete LFW dataset 
* **mixed_face_dataset_subset/**  
    Generated mixed masked and unmasked dataset from AFDB dataset, since too many unmasked pictures in AFDB dataset
* **preprocessing/**  
    Contains files to preprocess images, i.e. size and shit
* **saved_models/**  
    Saved models from transfer learning runs
* **team_pictures/**  
    Pictures of the team
* **team_pictures_cropped/**  
    Cropped pictures of the team
* **transfer_learning_triplet.py**
    transfer learning using triplet loss
* **transfer_learning.py**  
    transfer learning using classification
* **tripletdataset.py**  
    Contains Dataset class needed for transfer_learning_triplet
* **validate_lfw.py**  
    LFW validation code copied from pytorch-facenet
* **verify_all_pairs.py**  
    Count every single pair in mixed_face_dataset as a data point, generate validation statistics on our models
* **face_verification_one_shot.py**  
    Experiment with face verification on tensorflow
