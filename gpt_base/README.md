# The GPT-2 model based on the release of OpenAI

## data pre-process
python preprocess.py data_folder GPT_folder

## train
CUDA_VISIBLE_DEVICES=0,1 python Main.py --mode train

## test
CUDA_VISIBLE_DEVICES=0,1 python Main.py --mode test --saved_model_path load_model_path

