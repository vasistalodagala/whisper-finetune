import argparse
from transformers import pipeline

parser = argparse.ArgumentParser(description='Script to transcribe a custom audio file of any length using Whisper Models of various sizes.')
parser.add_argument(
    "--is_public_repo",
    required=False,
    default=True, 
    type=lambda x: (str(x).lower() == 'true'),
    help="If the model is available for download on huggingface.",
)
parser.add_argument(
    "--hf_model",
    type=str,
    required=False,
    default="openai/whisper-tiny",
    help="Huggingface model name. Example: openai/whisper-tiny",
)
parser.add_argument(
    "--ckpt_dir",
    type=str,
    required=False,
    default=".",
    help="Folder with the pytorch_model.bin file",
)
parser.add_argument(
    "--temp_ckpt_folder",
    type=str,
    required=False,
    default="temp_dir",
    help="Path to create a temporary folder containing the model and related files needed for inference",
)
parser.add_argument(
    "--path_to_audio",
    type=str,
    required=True,
    help="Path to the audio file to be transcribed.",
)
parser.add_argument(
    "--language",
    type=str,
    required=False,
    default="hi",
    help="Two letter language code for the transcription language, e.g. use 'hi' for Hindi. This helps initialize the tokenizer.",
)
parser.add_argument(
    "--device",
    type=int,
    required=False,
    default=0,
    help="The device to run the pipeline on. -1 for CPU, 0 for the first GPU (default) and so on.",
)

args = parser.parse_args()

if args.is_public_repo == False:
    os.system(f"mkdir -p {args.temp_ckpt_folder}")
    ckpt_dir_parent = str(Path(args.ckpt_dir).parent)
    os.system(f"cp {ckpt_dir_parent}/added_tokens.json {ckpt_dir_parent}/normalizer.json \
    {ckpt_dir_parent}/preprocessor_config.json {ckpt_dir_parent}/special_tokens_map.json \
    {ckpt_dir_parent}/tokenizer_config.json {ckpt_dir_parent}/merges.txt \
    {ckpt_dir_parent}/vocab.json {args.ckpt_dir}/config.json {args.ckpt_dir}/pytorch_model.bin \
    {args.ckpt_dir}/training_args.bin {args.temp_ckpt_folder}")
    model_id = args.temp_ckpt_folder
else:
    model_id = args.hf_model

transcribe = pipeline(
    task="automatic-speech-recognition",
    model=model_id,
    chunk_length_s=30,
    device=args.device,
)

transcribe.model.config.forced_decoder_ids = transcribe.tokenizer.get_decoder_prompt_ids(language=args.language, task="transcribe")
print('Transcription: ')
print(transcribe(args.path_to_audio)["text"])

if args.is_public_repo == False:
    os.system(f"rm -r {args.temp_ckpt_folder}")
