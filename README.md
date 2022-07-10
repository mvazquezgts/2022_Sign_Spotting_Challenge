# Pipeline of the baseline solution for MSSL and OSLWL tracks.
<img src="./img/SignSpottingDiagram.drawio.svg">

# Preparation: Main path, download code, download data.
```bash
PATH_FOLDER_REPOSITORY='/home/gts/projects/mvazquez' 
```

```bash
cd $PATH_FOLDER_REPOSITORY
git clone https://github.com/ManuelGTM/2022_Sign_Spotting_Challenge.git
```

```bash
cd $PATH_FOLDER_REPOSITORY/2022_Sign_Spotting_Challenge/CSLR_workspace/SignSpotting/data
sh script_download_data.sh 
```


# Running experiments
## TRACK1

```bash
cd $PATH_FOLDER_REPOSITORY/2022_Sign_Spotting_Challenge/CSLR_workspace/SignSpotting/src
python main.py --experiment MSSL/EXPERIMENTO_MSSL_TRAIN_SET --path_cslr $PATH_FOLDER_REPOSITORY'/2022_Sign_Spotting_Challenge/CSLR_workspace' --path_islr $PATH_FOLDER_REPOSITORY'/2022_Sign_Spotting_Challenge/ISLR_workspace'
python main.py --experiment MSSL/EXPERIMENTO_MSSL_VAL_SET --path_cslr $PATH_FOLDER_REPOSITORY'/2022_Sign_Spotting_Challenge/CSLR_workspace' --path_islr $PATH_FOLDER_REPOSITORY'/2022_Sign_Spotting_Challenge/ISLR_workspace'
python main.py --experiment MSSL/EXPERIMENTO_MSSL_TEST_SET --path_cslr $PATH_FOLDER_REPOSITORY'/2022_Sign_Spotting_Challenge/CSLR_workspace' --path_islr $PATH_FOLDER_REPOSITORY'/2022_Sign_Spotting_Challenge/ISLR_workspace'
```

## TRACK2
```bash
cd $PATH_FOLDER_REPOSITORY/2022_Sign_Spotting_Challenge/CSLR_workspace/SignSpotting/src
python main_step1_track2.py --experiment OSLWL/EXPERIMENTO_OSLWL_QUERY_VAL_SET --path_cslr $PATH_FOLDER_REPOSITORY'/2022_Sign_Spotting_Challenge/CSLR_workspace' --path_islr $PATH_FOLDER_REPOSITORY'/2022_Sign_Spotting_Challenge/ISLR_workspace'
python main_step1_track2.py --experiment OSLWL/EXPERIMENTO_OSLWL_VAL_SET --path_cslr $PATH_FOLDER_REPOSITORY'/2022_Sign_Spotting_Challenge/CSLR_workspace' --path_islr $PATH_FOLDER_REPOSITORY'/2022_Sign_Spotting_Challenge/ISLR_workspace'
python main_step2_track2.py --experiment OSLWL/EXPERIMENTO_OSLWL_VAL_SET --path_cslr $PATH_FOLDER_REPOSITORY'/2022_Sign_Spotting_Challenge/CSLR_workspace' --path_islr $PATH_FOLDER_REPOSITORY'/2022_Sign_Spotting_Challenge/ISLR_workspace'
```

Note:
By default, the scripts only show the commands ready to run, so the user must run them manually. But these scripts also implement a sequential and autonomous execution of all commands, for this purpose an argument called "--run" is included. Example:
```bash
python main.py --experiment MSSL/EXPERIMENTO_MSSL_TRAIN_SET --path_cslr $PATH_FOLDER_REPOSITORY'/2022_Sign_Spotting_Challenge/CSLR_workspace' --path_islr $PATH_FOLDER_REPOSITORY'/2022_Sign_Spotting_Challenge/ISLR_workspace' --run 
```

# Download the files generated in each experiment.

```bash
cd $PATH_FOLDER_REPOSITORY/2022_Sign_Spotting_Challenge/CSLR_workspace/SignSpotting/experiments
sh script_download_experiments.sh 
```

In each phase a series of files are generated that are necessary for the following phases. All these data are stored in folders, so we can run an experiment from any intermediate stage using the previously saved files.


# Run a new experiment
1. Put in the "/data" folder all the videos and .eaf files (if necessary).
2. Create a configuration file in "src/config". This file contains all the necessary information related to an experiment in order to execute all the commands . It is recommended starting from a configuration file already available for each track and edit it.
