# Fine-tuning and evaluating Whisper models for Automatic Speech Recognition

This repository contains the relevant scripts to fine-tune and evaluate Whisper models of various configurations available over huggingface 🤗.

Scripts in this repository support fine-tuning these models using custom datasets which haven't been made available over huggingface.
Some of the models trained and evaluated using these scripts can be found [here on huggingface](https://huggingface.co/vasista22).

Code snippets have been made available to extract relevant embeddings from different layers of whisper models with varied configurations.


## Contents

- [Setup](#setup)
- [Data Preparation for custom datasets](#data-preparation-for-custom-datasets)
- [Hyperparameter tuning](#hyperparameter-tuning)
- [Fine-tune on a dataset from huggingface](#fine-tune-on-a-dataset-from-huggingface)
- [Fine-tune on a custom dataset](#fine-tune-on-a-custom-dataset)
- [Evaluate on a dataset from huggingface](#evaluate-on-a-dataset-from-huggingface)
- [Evaluate on a custom dataset](#evaluate-on-a-custom-dataset)
- [Transcribe a single audio file](#transcribe-a-single-audio-file)
- [Faster evaluation with whisper-jax](#faster-evaluation-with-whisper-jax)
- [Extract embeddings from whisper models](#extract-embeddings-from-whisper-models)
- [Interesting works around Whisper](#interesting-works-around-whisper)


## Setup

These scripts have been tested with Python 3.8 and cuda 11.3.

It is recommended that you setup a virtual environment for the installation purpose and work within the same. The following set of commands would setup a virtual environment and complete the installation:

```bash
python3 -m venv env_whisper-finetune
source env_whisper-finetune/bin/activate

python3 -m pip install --upgrade pip
pip install -r requirements.txt
```

In order to push your model to huggingface, you would need to login using the command line interface. Also, `git-lfs` would need to be installed to push large model files. Executing the following commands should help in this regard:
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

# source_data_directory is the path to the directory containing the `text` and `audio_paths` files
# output_data_directory is where the formatted data would be stored

python3 custom_data/data_prep.py \
--source_data_dir source_data_directory \
--output_data_dir output_data_directory
```

Use the `python3 custom_data/data_prep.py -h` command for further detail on its usage.

## Hyperparameter tuning

Learning rate is one of the most important hyperparameters while trying to adapt/fine-tune models, and more so with models such as Whisper which have been pre-trained on vast amounts of data.

According to Jong Wook Kim, one of the authors of the Whisper paper, a practical learning rate to consider while fine-tuning is a value that is 40x smaller than what has been used for pre-training, and linearly decay it to zero over the course of training. ([Discord thread where this has been mentioned](https://discord.com/channels/879548962464493619/1050020275250548836/1050369079111856168))

The following table contains the suggested learning rates for the different model configurations for the fine-tuning experiments:

| Model Size | Max Learning Rate (paper) | Suggested fine-tuning Learning Rate (40x smaller) |
|   :---:    |           :---:           |                      :---:                        |
|   tiny     |      $1.5$ x $10^{-3}$    |                  $3.75$ x $10^{-5}$               |
|   base     |      $1$ x $10^{-3}$      |                  $2.5$ x $10^{-5}$                |
|   small    |      $5$ x $10^{-4}$      |                  $1.25$ x $10^{-5}$               |
|   medium   |      $2.5$ x $10^{-4}$    |                  $6.25$ x $10^{-6}$               |
|   large    |      $1.75$ x $10^{-4}$   |                  $4.375$ x $10^{-6}$              |
|   large-v2 |      $2.0$ x $10^{-4}$    |                  $5$ x $10^{-6}$                  |

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

**NOTE:** whisper-jax can only be used on those models which also have their flax weights available over huggingface. To push the flax weights for existing models, one may follow the instructions given [here](https://github.com/sanchit-gandhi/whisper-jax#available-models-and-languages).

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

## Extract embeddings from whisper models

Given the enormous amount of speech data that Whisper models have been trained on, embeddings from these models (original/fine-tuned) can also be used for other speech downstream tasks apart from Automatic Speech Recognition (ASR).

The following table contains the dimensions of the encoder and decoder embeddings for different model sizes:
| Model Size | Embedding Dimension | Number of Layers |
|   :---:    |        :---:        |       :---:      |
|   tiny     |        384          |         4        |
|   base     |        512          |         6        |
|   small    |        768          |        12        |
|   medium   |        1024         |        24        |
|   large    |        1280         |        32        |
|   large-v2 |        1280         |        32        |

The different embeddings available from the whisper Seq2Seq model output are:
- `encoder_last_hidden_state` - The output of the last layer of the encoder post layer norm.
- `encoder_hidden_states` - List of embeddings from every layer of the encoder. For example, the whisper tiny model would have 5 embeddings in this list. The indices 0 to 3 in this list would be the embeddings from the layer-1 to layer-4 of the encoder. The index-4 in this list, which is the 5-th embedding is same as `encoder_last_hidden_state`. That is, it corresponds to the final encoder layer's embedding after a layer-norm is applied.
- `last_hidden_state` - The output of the last layer of the decoder post layer norm.
- `decoder_hidden_states` - List of embeddings from every layer of the decoder. For example, the whisper tiny model would have 5 embeddings in this list. The indices 0 to 3 in this list would be the embeddings from the layer-1 to layer-4 of the decoder. The index-4 in this list, which is the 5-th embedding is same as `last_hidden_state`. That is, it corresponds to the final decoder layer's embedding after a layer-norm is applied.

The embeddings from the encoder could be used for downstream tasks such as Speaker Verification, Speaker Diarization, Speech Enhancement etc., where the speaker related information is more relevant.

When it comes to downstream tasks such as Keyword Spotting, Phoneme Recognition etc., which have more to do with the semantics of the data, the embeddings from the decoder could help better.

The following code snippet can be used to extract the different embeddings discussed above.

**NOTE:**
- Ensure that the audio segment being passed is no longer than 30 seconds in duration. This is because whisper's positional embeddings etc., are designed to handle speech segments that are atmost 30 seconds in duration. The features from longer audios are truncated and the features from shorter ones are padded. The [WhisperConfig](https://github.com/huggingface/transformers/blob/main/src/transformers/models/whisper/configuration_whisper.py#L62) class specifies in the definition of `max_source_positions` argument that `1500` is 'The maximum sequence length of log-mel filter-bank features that this model might ever be used with.' This in terms of time duration coressponds to 30 seconds.
- The mean of the embeddings of any layer can be used to represent that particular layer's output for the audio segment through a single embedding.

```python

import torch
from datasets import Dataset, Audio
from transformers import AutoFeatureExtractor, WhisperModel

audio_segment_path="/path/to/the/audio_file"  # pass the path to the audio segment (<= 30 seconds) here.

model = WhisperModel.from_pretrained("vasista22/whisper-kannada-small")  # The model ID to use can be changed here
feature_extractor = AutoFeatureExtractor.from_pretrained("vasista22/whisper-kannada-small")  # The model ID to use can be changed here
model.eval()

# creating a pseudo dataset to extract features for the audio segment
audio_read = Dataset.from_dict({"audio": [audio_segment_path]}).cast_column("audio", Audio(sampling_rate=16_000))
inputs = feature_extractor(audio_read['audio'][0]['array'], sampling_rate=16_000, return_tensors="pt")
input_features = inputs.input_features

model.config.output_hidden_states=True  # to obtain the individual layer embeddings
decoder_input_ids = torch.tensor([[1, 1]]) * model.config.decoder_start_token_id

whisper_embeddings = model(input_features, decoder_input_ids=decoder_input_ids)

print('\n Last layer embeddings from whisper encoder post layer-norm: ', whisper_embeddings.encoder_last_hidden_state)
print('\n Mean of last layer embeddings from whisper encoder post layer-norm: ', torch.mean(whisper_embeddings.encoder_last_hidden_state, dim=1))
print('\n Embeddings from the 8-th encoder layer: ', whisper_embeddings.encoder_hidden_states[7])
print('\n Mean of the embeddings from the 8-th encoder layer: ', torch.mean(whisper_embeddings.encoder_hidden_states[7], dim=1))
print('\n Last layer embeddings of whisper decoder post layer-norm: ', whisper_embeddings.last_hidden_state)
print('\n Mean of last layer embeddings from whisper decoder post layer-norm: ', torch.mean(whisper_embeddings.last_hidden_state, dim=1))
print('\n Embeddings from the 8-th decoder layer: ', whisper_embeddings.decoder_hidden_states[7])
print('\n Mean of the embeddings from the 8-th decoder layer: ', torch.mean(whisper_embeddings.decoder_hidden_states[7], dim=1))

```

## Interesting works around Whisper

Since the release of Whisper models and code from OpenAI, there have been several developments in bringing out and enhancing the capabilities of these models. Following are few such works which could potentially be of some use to researchers and developers:

- Efficient Inference
    - [whisper-jax](https://github.com/sanchit-gandhi/whisper-jax)
    - [faster-whisper](https://github.com/guillaumekln/faster-whisper)
    - [whisper.cpp](https://github.com/ggerganov/whisper.cpp)
- Accurate Timestamps
    - [whisper-timestamped](https://github.com/linto-ai/whisper-timestamped)
    - [stable-ts](https://github.com/jianfch/stable-ts)
- Forced Alignment using an external Phoneme based ASR model
    - [whisperX](https://github.com/m-bain/whisperX)
- Parameter Efficient Fine-tuning
    - [fast-whisper-finetuning](https://github.com/Vaibhavs10/fast-whisper-finetuning)