# Logic2Text

## Data
In the dataset folder, we have the full dataset (all_data.json), and the train test split (train.json, valid.json, test.json). 
Each example is in a dictionary of the following format:
```
    "topic": table caption,
    "wiki": link to the wikipedia page,
    "action": logic type,
    "sent": description,
    "annotation": raw annotation data from the mturk,
    "logic": logic form,
    "logic_str": linearized logic form,
    "nid": number of fuction nodes,
    "table_header": table header, in list format
    "table_cont": table content, list of lists
  
```

## Logic form execution
In the execution folder, run
```
python execute.py
```
It will execute all the logic forms in all_data.json. All the function definitions are in APIs.py

This site is under construction, and we will release other codes in the future.

## Model
The pre-trained GPT-2 can be downloaded via [Dropbox](https://www.dropbox.com/sh/99awpjnj2lh4e17/AACCz_XU_FhkinSId0_nz1-qa?dl=0).

### data pre-process
In addition to the original data folder, this script will create another folder to contain the preprocessed data, which will be used for train and test.

python preprocess.py data_folder GPT_folder

### train
Modify the data paths and parameters in the Main.py file. Then run:

CUDA_VISIBLE_DEVICES=0,1 python Main.py --mode train

### test
CUDA_VISIBLE_DEVICES=0,1 python Main.py --mode test --saved_model_path load_model_path
