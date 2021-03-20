# Overfitting of the iterative Pierrick's Model


## REQUERIMIENTS

```bash
conda install pytorch=1.7.1 torchvision=0.8.2 cudatoolkit=11 -c pytorch
pip install -r requirements.txt
```

## Preparing datasets


Datasets are created from a set of image frames. If you have a video, the video has to be first transformed into a sequence of frame images. This can be done using ffmpeg 
utils
```bash
ffmpeg -i video_a.mp4 ./video_frames/tennis/frames/%05d.jpg -hide_banner
```
A mask image must be manually created. A sample mask can be 
found in ./video_frames/mask.png

The datasets are composed by the inputs to the net and the ground truth. 
- inputs
  - The masked optical flow (forward and backward)
  - The masks 
- The ground truth
  - the ground truth optical flow (forward and backward)
    
The script `create_dataset.py` also creates additional files not needed (yet).

- The Ground Truth frames (the non-masked frames)

In order to create the dataset from a set of image frames (in the repository you have a already created one 
in video_frames/tennis),, runs the following script 

```bash
cd training
python create_dataset.py --in_root_dir ../video_frames --out_dir ../dataset  --masking_mode template --template_mask ../video_frames/tennis/mask.png --compute_RAFT_flow --apply_mask_before
```

## Running the  overfitting training over data in `./datasets`


```bash
cd training
python launch_training.py
```

The script `launch_training.py` train first the encoder/decoder and once is converged, the update net is trained.
The parameters of the training are defined at the beginning of the file

### Architecture and losses
The architecture and losses are described in file `./doc/Flow_inpainting.pdf` chapter 3, page 18