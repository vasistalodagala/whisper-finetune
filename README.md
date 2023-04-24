# Fine-tuning and evaluating Whisper models for Automatic Speech Recognition

This repository contains the relevant scripts to fine-tune and evaluate Whisper models of various configurations available over huggingface 🤗.
There is support to fine-tune the models using custom datasets that haven't been made available over huggingface.
Some of the models trained and evaluated using these scripts can be found [here on huggingface](https://huggingface.co/vasista22).


## Table of Contents

- [Setup](#setup)
- [Data Preparation for custom datasets](#data-preparation-for-custom-datasets)
- [Fine-tune on a dataset from huggingface](#fine-tune-on-a-dataset-from-huggingface)
- [Fine-tune on a custom dataset](#fine-tune-on-a-custom-dataset)
- [Evaluate on a dataset from huggingface](#evaluate-on-a-dataset-from-huggingface)
- [Evaluate on a custom dataset](#evaluate-on-a-custom-dataset)
- [Transcribe a single audio file](#transcribe-a-single-audio-file)
- [Faster evaluation with whisper-jax](#faster-evaluation-with-whisper-jax)
- [Interesting works around Whisper](#interesting-works-around-whisper)


## Setup ⚙️

These scripts have been tested with Python 3.8 and cuda 11.3.

It is recommended that you setup a virtual environment for the installation purpose and work within the same. The following set of commands would setup a virtual environment and complete the installation:

```bash
python3 -m venv env_whisper-finetune
source env_whisper-finetune/bin/activate

python3 -m pip install --upgrade pip
pip install -r requirements.txt
```

In order to push your model to huggingface, you would need to login using the command line interface. Also, `git-lfs` would need to be installed to push large model files. The following commands should help in this regard:
```bash
sudo apt-get install git-lfs
huggingface-cli login
```


## Data Preparation for custom datasets

**NOTE:** The contents of this section are relevant only if you are interested in using dataset(s) that aren't available over huggingface. You may proceed to the later sections of this README if this isn't applicable to your usecase.

One could be interested in working with a dataset that isn't available over huggingface.
To fine-tune whisper models or evaluate them on such datasets, a preliminary data preparation is needed to make them compatible with the huggingface's sequence-to-sequence training pipeline.

The script which converts the dataset into the required format, expects two files named `text` and `audio_paths`.

The `audio_paths` file is expected to contain the absolute paths to each of the audio files to be used in the fine-tuning or evaluation process. Also, each entry in the file has to be indexed by a unique utterance ID. The contents of the file should be organized in the following manner.
```bash
<unique-id> <absolute path to the audio file-1>
<unique-id> <absolute path to the audio file-2>
...
<unique-id> <absolute path to the audio file-N>
```

The `text` file is expected to contain the transcriptions corresponding to each of the audio files mentioned in the `audio_paths` file. Also, each entry in the file has to be indexed by a unique utterance ID. The ordering of unique utterance IDs in both the `text` and `audio_paths` files should be consistent. The contents of the `text` file should be organized in the following manner.
```bash
<unique-id> <Transcription (ground truth) corresponding to the audio file-1>
<unique-id> <Transcription (ground truth) corresponding to the audio file-2>
...
<unique-id> <Transcription (ground truth) corresponding to the audio file-N>
```

The `sample_data` folder of this repository provides a reference on how these two files are to be organized.

Once the data has been organized in the manner, the script named `custom_data/data_prep.py` could be used to convert the data into the format expected by sequence-to-sequence pipeline of huggingface. 

Following is a sample command to convert the data into the desired format:

```bash
python3 custom_data/data_prep.py \
--source_data_dir source_data_directory \
--output_data_dir output_data_directory
```

Use the `python3 custom_data/data_prep.py -h` command for further detail on its usage.


## Fine-tune on a dataset from huggingface

To fine-tune a Whisper model on a dataset available over huggingface, the `train/fine-tune_on_hf_dataset.py` file can be used.

Following is a sample command to perform the same:

```bash
ngpu=4  # number of GPUs to perform distributed training on.

torchrun --nproc_per_node=${ngpu} train/fine-tune_on_hf_dataset.py \
--model_name vasista22/whisper-hindi-base \
--language Hindi \
--sampling_rate 16000 \
--num_proc 2 \
--train_strategy steps \
--learning_rate 3e-3 \
--warmup 1000 \
--train_batchsize 16 \
--eval_batchsize 8 \
--num_steps 10000 \
--resume_from_ckpt None \
--output_dir op_dir_steps \
--train_datasets mozilla-foundation/common_voice_11_0 mozilla-foundation/common_voice_11_0 \
--train_dataset_configs hi hi \
--train_dataset_splits train validation \
--train_dataset_text_columns sentence sentence \
--eval_datasets "google/fleurs" \
--eval_dataset_configs hi_in \
--eval_dataset_splits test \
--eval_dataset_text_columns transcription
```

Multiple datasets can be used as a part of the fine-tuning process. These datasets would be concatenated and shuffled at the time of dataset preparation.
It is to be noted that the number of paramenters passed through the `train_datasets`, `train_dataset_configs`, `train_dataset_splits` and `train_dataset_text_columns` arguments should be the same and the ordering of parameters between these arguments should be consistent. The same applies to the `eval_datasets`, `eval_dataset_configs`, `eval_dataset_splits` and `eval_dataset_text_columns` arguments.

Use the `python3 train/fine-tune_on_hf_dataset.py -h` command for further detail on its usage.

While all of the arguments are set with default options, one is encouraged to look into the file to customize the training hyperparameters in such a way that it suits the amount of data at hand and the size of the model being used.

## Fine-tune on a custom dataset

To fine-tune a Whisper model on a custom dataset, the `train/fine-tune_on_custom_dataset.py` file can be used.

Following is a sample command to perform the same:

```bash
ngpu=4  # number of GPUs to perform distributed training on.

torchrun --nproc_per_node=${ngpu} train/fine-tune_on_custom_dataset.py \
--model_name vasista22/whisper-telugu-base \
--language Telugu \
--sampling_rate 16000 \
--num_proc 2 \
--train_strategy epoch \
--learning_rate 3e-3 \
--warmup 1000 \
--train_batchsize 16 \
--eval_batchsize 8 \
--num_epochs 20 \
--resume_from_ckpt None \
--output_dir op_dir_epoch \
--train_datasets output_data_directory/train_dataset_1 output_data_directory/train_dataset_2 \
--eval_datasets output_data_directory/eval_dataset_1 output_data_directory/eval_dataset_2 output_data_directory/eval_dataset_3
```

The datasets being passed as parameters through the `train_datasets` and `eval_datasets` arguments should have been from the output directories generated through the data preparation stage.
Multiple datasets can be used as a part of the fine-tuning process. These datasets would be concatenated and shuffled at the time of dataset preparation.

Use the `python3 train/fine-tune_on_custom_dataset.py -h` command for further detail on its usage.

While all of the arguments are set with default options, one is encouraged to look into the file to customize the training hyperparameters in such a way that it suits the amount of data at hand and the size of the model being used.


## Evaluate on a dataset from huggingface

The `evaluate/evaluate_on_hf_dataset.py` file can be used to evaluate models on a dataset available over huggingface. The model to be evaluated however, can either be a Whisper model from huggingface or a local Whisper checkpoint generated during the fine-tuning stage.

Following is a sample command to perform the same:

```bash
python3 evaluate/evaluate_on_hf_dataset.py \
--is_public_repo False \
--ckpt_dir "op_dir_epoch/checkpoint-394" \
--temp_ckpt_folder "temp" \
--language gu \
--dataset "google/fleurs" \
--config gu_in \
--split test \
--device 0 \
--batch_size 16 \
--output_dir predictions_dir
```

The `is_public_repo` argument takes in a boolean value and it specifies whether the model to evaluate is a model from huggingface or if it is a local checkpoint. The above command would evaluate the local checkpoint on a dataset from huggingface. Also, the `ckpt_dir` and `temp_ckpt_folder` arguments are relevant only when evaluating a local checkpoint.

To evaluate a model from huggingface, `is_public_repo` should be set to `True` and the model id should be passed through the `hf_model` argument. The following is a sample command to perform the same:

```bash
python3 evaluate/evaluate_on_hf_dataset.py \
--is_public_repo True \
--hf_model vasista22/whisper-kannada-small \
--language kn \
--dataset "google/fleurs" \
--config kn_in \
--split test \
--device 0 \
--batch_size 16 \
--output_dir predictions_dir
```

After succesful execution `--output_dir` would be containing one result file per dataset which would contain the word error rate and character error rate results along with the reference (REF) for each utterance in the dataset and the hypothesis (HYP) generated by the model. These result files would be named based on the name of the model and the name of dataset being evaluated on.

Use the `python3 evaluate/evaluate_on_hf_dataset.py -h` command for further detail on its usage.

While all of the arguments are set with default options, one is encouraged to look into the file to customize the arguments. For instance, CPU inference would require the `device` argument to be set to `-1`.


## Evaluate on a custom dataset

The `evaluate/evaluate_on_custom_dataset.py` file can be used to evaluate models on a custom dataset prepared using the data preparation stage described above. The model to be evaluated however, can either be a Whisper model from huggingface or a local Whisper checkpoint generated during the fine-tuning stage.

Following is a sample command to perform the same:

```bash
python3 evaluate/evaluate_on_custom_dataset.py \
--is_public_repo False \
--ckpt_dir "op_dir_epoch/checkpoint-394" \
--temp_ckpt_folder "temp" \
--language gu \
--eval_datasets output_data_directory/eval_dataset_1 output_data_directory/eval_dataset_2 \
--device 0 \
--batch_size 16 \
--output_dir predictions_dir
```

The model can be evaluated on multiple datasets and they can be passed as mentioned in the above command. The results on each of these datasets would be stored in individual files in the `--output_dir`.

The `is_public_repo` argument takes in a boolean value and it specifies whether the model to evaluate is a model from huggingface or if it is a local checkpoint. The above command would evaluate the local checkpoint on a dataset from huggingface. Also, the `ckpt_dir` and `temp_ckpt_folder` arguments are relevant only when evaluating a local checkpoint.

To evaluate a model from huggingface, `is_public_repo` should be set to `True` and the model id should be passed through the `hf_model` argument. The following is a sample command to perform the same:

```bash
python3 evaluate/evaluate_on_custom_dataset.py \
--is_public_repo True \
--hf_model vasista22/whisper-kannada-small \
--language kn \
--eval_datasets output_data_directory/eval_dataset_1 output_data_directory/eval_dataset_2 \
--device 0 \
--batch_size 16 \
--output_dir predictions_dir
```

After succesful execution `--output_dir` would be containing one result file per dataset which would contain the word error rate and character error rate results along with the reference (REF) for each utterance in the dataset and the hypothesis (HYP) generated by the model. These result files would be named based on the name of the model and the name of dataset being evaluated on.

Use the `python3 evaluate/evaluate_on_custom_dataset.py -h` command for further detail on its usage.

While all of the arguments are set with default options, one is encouraged to look into the file to customize the arguments. For instance, CPU inference would require the `device` argument to be set to `-1`.


## Transcribe a single audio file

The `transcribe_audio.py` file can be used to obtain the transcription of a single audio file. The model being used for the transcription can either be a Whisper model from huggingface or a local Whisper checkpoint generated during the fine-tuning stage.

Following is a sample command to perform the same:

```bash
python3 transcribe_audio.py \
--is_public_repo False \
--ckpt_dir "op_dir_epoch/checkpoint-1254" \
--temp_ckpt_folder "temp" \
--path_to_audio /path/to/audio/file.wav \
--language ta \
--device 0
```

The `is_public_repo` argument takes in a boolean value and it specifies whether the model to be used is a model from huggingface or if it is a local checkpoint. The above command would transcribe the audio using a local checkpoint. Also, the `ckpt_dir` and `temp_ckpt_folder` arguments are relevant only when using a local checkpoint.

To make use of a model from huggingface, `is_public_repo` should be set to `True` and the model id should be passed through the `hf_model` argument. The following is a sample command to perform the same:

```bash
python3 transcribe_audio.py \
--is_public_repo True \
--hf_model vasista22/whisper-tamil-base \
--path_to_audio /path/to/audio/file.wav \
--language ta \
--device 0
```

Use the `python3 transcribe_audio.py -h` command for further detail on its usage.

While most of the arguments are set with default options, one is encouraged to look into the file to customize the arguments. For instance, CPU inference would require the `device` argument to be set to `-1`.


## Faster evaluation with whisper-jax

[whisper-jax](https://github.com/sanchit-gandhi/whisper-jax) helps speed up the inference of whisper models. The `evaluate/jax_evaluate_on_hf_dataset.py` and `evaluate/jax_evaluate_on_custom_dataset.py` files make use of whisper-jax to speed up evaluation on datasets from huggingface and custom datasets respectively.

In order to make use of this faster evaluation, please install the necessary dependencies as suggested in the [whisper-jax](https://github.com/sanchit-gandhi/whisper-jax) repository. If you're using CUDA 11, the following commands should safely complete the installation:
```bash
pip install --upgrade pip
pip install --upgrade "jax[cpu]"  # cpu installation of jax
pip install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html  # jax for gpu

pip install git+https://github.com/sanchit-gandhi/whisper-jax.git
```

whisper-jax can only be used on those models which also have their flax weights available over huggingface. To push the flax weights for existing models, one may follow the instructions given [here](https://github.com/sanchit-gandhi/whisper-jax#available-models-and-languages).

Following is a sample command to evaluate the model on a dataset from huggingface:

```bash
python3 evaluate/jax_evaluate_on_hf_dataset.py \
--hf_model vasista22/whisper-telugu-small \
--language te \
--dataset "google/fleurs" \
--config te_in \
--split test \
--device 0 \
--batch_size 16 \
--output_dir jax_predictions_dir \
--half_precision True
```

Similarly following is a sample command to evaluate the model on a custom dataset:

```bash
python3 evaluate/jax_evaluate_on_custom_dataset.py \
--hf_model openai/whisper-base \
--language hi \
--eval_datasets output_data_directory/eval_dataset_1 output_data_directory/eval_dataset_2 \
--device 0 \
--batch_size 16 \
--output_dir jax_predictions_dir \
--half_precision True
```

The model computation can be run in half-precision by setting the `--half_precision` argument to `True`. This helps further speed up the computations.

While running inference using whisper-jax, if you are facing an error message that reads `Failed to determine best cudnn convolution algorithm/No GPU/TPU found`, a possible solution [suggested](https://github.com/google/jax/issues/8746#issuecomment-1327919319) is to export the following commands:

```bash
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_FLAGS="--xla_gpu_strict_conv_algorithm_picker=false --xla_gpu_force_compilation_parallelism=1"
```

To transcribe a single audio file using whisper-jax, the `jax_transcribe_audio.py` command can be used. Following is a sample command regarding its usage:

```bash
python3 jax_transcribe_audio.py \
--hf_model vasista22/whisper-tamil-base \
--path_to_audio /path/to/audio/file.wav \
--language ta \
--device 0 \
--half_precision True \
--batch_size 16
```

## Interesting works around Whisper

Since the release of Whisper models and code from OpenAI, there have been several works that have worked on bringingout and enhancing the capabilities of these models. Following are few such works which can potentially be of some use to developers:

- Efficient Inference
    - [whisper-jax](https://github.com/sanchit-gandhi/whisper-jax)
    - [faster-whisper](https://github.com/guillaumekln/faster-whisper)
    - [whisper.cpp](https://github.com/ggerganov/whisper.cpp)
- Accurate Timestamps
    - [whisper-timestamped](https://github.com/linto-ai/whisper-timestamped)
    - [stable-ts](https://github.com/jianfch/stable-ts)
- Forced Alignment using an external Phoneme based ASR model
    - [whisperX](https://github.com/m-bain/whisperX)