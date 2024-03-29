## REQUERIMIENTS

```bash
conda create --name VideoInp python=3.7.10
conda activate VideoInp
#adding channel nvidia to install exact version of cudatoolkit
conda config --add channels nvidia
conda install pytorch=1.7.1 torchvision=0.8.2 cudatoolkit=11.2.2 -c pytorch
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

In order to create the dataset from a set of image frames (in the repository you have a already created a set of images 
in video_frames/tennis), runs the following script 

```bash
cd ingestion
`_python create_dataset.py --in_root_dir ../video_frames --out_dir ../dataset  --masking_mode same_template --template_mask ../video_frames/tennis/mask.png --apply_mask_before --H 256 --W 480 --nLevels 2_`
```

other examples for creating datasets:

```bash
python create_dataset.py --in_root_dir ../../../data/datasets/raw/davis_no_mask/ --out_dir ../../../data/datasets/built/davis_noMask_multiscale_3_B  --masking_mode same_template --template_mask /home/gpi/workspace/data/datasets/mask_templates/no_mask.png   --H 256 --W 480 --nLevels 3
```

## Running the  overfitting training over data in `./datasets`

Change file ./config to rename the experiment and change  '"random_holes_on_the_fly": true,' to false if you want to tarain with the 
mask

In the root dir
```bash
python main.py ./configs/example.json 
```

### Architecture and losses
The architecture and losses are described in file `./doc/Flow_inpainting.pdf` chapter 3, page 18

### Streamlit App
In order to compare two different models use the streamlit app

```bash
cd UI_verbose
streamlit run streamlit_app.py ../verbose/training_out/
```

Streamlit_app reads from '../verbose/training_out/' the output of the trainings and shows in a 
web.

