# VideoInp

conda install pytorch torchvision cudatoolkit=11.0 -c pytorch
conda install matplotlib scipy
pip install -r requirements.txt


## How to check if the encoder-decoder pipeline is working well
```bash
check_encoder_decoder.py
``` 

Parameters
* **--saving_path _path_**: Root folder where the results will be saved
* **--flow2features_weights _path_**: Path to the weights of the  encoder (flow to features) network architecture
* **--features2flow_weights _path_**: Path to the weights of the  decoder (features to flow) network architecture
* **--video_path _path_**: Path where the video frames are located. Inside this folder, should exist a folder called 'frames' containing the frames
* **--features_perturbation** : If this flag is activated, random noise is added to the result of the encoder and previous to the decoder.
* **--opticalFlow_model _path_**: Path to the RAFT checkpoint for computing the Optical flow.

###Example
* Check without perturbation
```bash
python check_encoder_decoder.py --video_path ../data/tennis  
                                --saving_path ../check_enc_dec_borrar 
                                --flow2features_weights ../weight/inpModel/iterative_inpainting_single_decoding_2_pow_update_5frames_no_schedule_fill/ckpt/flow_2F_19999.pth 
                                --features2flow_weights  ../weight/inpModel/iterative_inpainting_single_decoding_2_pow_update_5frames_no_schedule_fill/ckpt/F2flow_19999.pth
```

* Check with perturbation
```bash
python check_encoder_decoder.py --video_path ../data/tennis  
                                --saving_path ../check_enc_dec_borrar 
                                --flow2features_weights ../weight/inpModel/iterative_inpainting_single_decoding_2_pow_update_5frames_no_schedule_fill/ckpt/flow_2F_19999.pth 
                                --features2flow_weights  ../weight/inpModel/iterative_inpainting_single_decoding_2_pow_update_5frames_no_schedule_fill/ckpt/F2flow_19999.pth
                                --features_perturbation
```