# Running the Iterative Pierrick's Checkpoints
**THIS VERSION ONLY INPAINTS THE OPTICAL FLOW (FROM A SET OF FRAME IMAGES AND MASKS), IT DOES NOT INPAINT THE FRAMES**

## REQUERIMIENTS

```bash
conda install pytorch=1.7.1 torchvision=0.8.2 cudatoolkit=11 -c pytorch
pip install -r requirements.txt
``` 

## How to Run an inference
From ./tools/ use

```bash
video_completion.py
``` 

Parameters

* **--flow2features_weights _path_** :  Path to the weights of the  encoder (flow to features) network architecture
* **--features2flow_weights _path_** : Path to the weights of the  decoder (features to flow) network architecture
* **--update_weights _path_** : Path to the weights of the update network architecture
* **--update_number_model _model_number_** : Class number of the update network architecture
* **--kind_update _pow | pol_** : Kind of update, by power or by polinomial
* **--mode _object_removal | video_extrapolation_** : Remove an object from the video, or extend the size. **ONLY IMPLEMENTED object_removal** 
* **--seamless** : Whether operate in the gradient domain when copy values.
* **--video_path _path_** : Path where the video frames are saved
* **--mask_path _path_** : Path where the video masks are saved
* **--outroot _path_** : output directory
* **--verbose** : If activated, save all intermediate steps
* **--verbose_path _path_** : where intermediate results will be saved
* **--opticalFlow_model _path_** : Path to the RAFT checkpoint for computing the Optical flow.

Example
```bash
python video_completion.py --verbose 
                           --verbose_path ../intermediate_results/
                           --mode object_removal 
                           --video_path ../data/tennis 
                           --mask_path ../data/tennis_mask 
                           --outroot ../result/tennis_removal 
                           --seamless  
                           --flow2features_weights ../weight/inpModel/iterative_inpainting_single_decoding_3_pow_update_5frames/ckpt/flow_2F_39999.pth
                           --features2flow_weights ../weight/inpModel/iterative_inpainting_single_decoding_3_pow_update_5frames/ckpt/F2flow_39999.pth
                           --update_weights ../weight/inpModel/iterative_inpainting_single_decoding_3_pow_update_5frames/ckpt/update_39999.pth
                           --update_number_model 2 
                           --kind_update pow
                           --opticalFlow_model ../weight/raft-things.pth
``` 


## How to check all checkpoints over a given video sequence

From ./tools/ use
```bash
test_pierrick_iterative_checkpoints.py
``` 

Parameters to use **from** command line
* **--chk_root_path _path_** : Root folder where the checkpoints (each one in a different folder inside root) are saved
* **--verbose** : If activated, save all intermediate steps
* **--verbose_path _path_** : Where intermediate results will be saved. Used joint to --verbose
* **--video_path _path_** : Path where the video frames are saved
* **--mask_path _path_** : Path where the video masks are saved
   
### Example

```bash
python test_pierrick_iterative_checkpoints.py --chk_root_path ../weight/inpModel/ 
                                              --verbose 
                                              --verbose_path ../experimet_results/
                                              --video_path ../data/tennis/  
                                              --mask_path ../data/tennis_mask/
``` 

## How to check if the encoder-decoder pipeline is working well
From ./tools/ use 
```bash
check_encoder_decoder.py
``` 

Parameters
* **--saving_path _path_**: Root folder where the results will be saved
* **--flow2features_weights _path_**: Path to the weights of the  encoder (flow to features) network architecture
* **--features2flow_weights _path_**: Path to the weights of the  decoder (features to flow) network architecture
* **--video_path _path_**: Path where the video frames are located.
* **--features_perturbation** : If this flag is activated, random noise is added to the result of the encoder (previous to the decoder).
* **--opticalFlow_model _path_**: Path to the RAFT checkpoint for computing the Optical flow.

### Example
* Check without perturbation
```bash
python check_encoder_decoder.py --video_path ../data/tennis/frames  
                                --saving_path ../check_enc_dec_borrar 
                                --flow2features_weights ../weight/inpModel/iterative_inpainting_single_decoding_2_pow_update_5frames_no_schedule_fill/ckpt/flow_2F_19999.pth 
                                --features2flow_weights  ../weight/inpModel/iterative_inpainting_single_decoding_2_pow_update_5frames_no_schedule_fill/ckpt/F2flow_19999.pth
```

* Check with perturbation
```bash
python check_encoder_decoder.py --video_path ../data/tennis/frames  
                                --saving_path ../check_enc_dec
                                --flow2features_weights ../weight/inpModel/iterative_inpainting_single_decoding_2_pow_update_5frames_no_schedule_fill/ckpt/flow_2F_19999.pth 
                                --features2flow_weights  ../weight/inpModel/iterative_inpainting_single_decoding_2_pow_update_5frames_no_schedule_fill/ckpt/F2flow_19999.pth
                                --features_perturbation
```

## Version History

In this section, a brief description of what is each version (labels) in the master branch has to be done

* **V1.0**: This version runs the Pierrick's checkpoints
