import argparse
import jax.numpy as jnp
from whisper_jax import FlaxWhisperForConditionalGeneration, FlaxWhisperPipline

parser = argparse.ArgumentParser(description='Script to transcribe a custom audio file of any length using Whisper Models of various sizes.')
parser.add_argument(
    "--hf_model",
    type=str,
    required=False,
    default="openai/whisper-tiny",
    help="Huggingface model name. Example: openai/whisper-tiny",
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
parser.add_argument(
    "--half_precision",
    required=False,
    default=False, 
    type=lambda x: (str(x).lower() == 'true'),
    help="Run with half precision.",
)
parser.add_argument(
    "--batch_size",
    type=int,
    required=False,
    default=16,
    help="Batch size for inference.",
)

args = parser.parse_args()

model_id = args.hf_model

if args.half_precision == False:
    transcribe = FlaxWhisperPipline(
        model_id,
        batch_size=args.batch_size
    )
else:
    transcribe = FlaxWhisperPipline(
        model_id,
        dtype=jnp.float16,
        batch_size=args.batch_size
    )

transcribe.model.config.forced_decoder_ids = transcribe.tokenizer.get_decoder_prompt_ids(language=args.language, task="transcribe")
print('Transcription: ')
print(transcribe(args.path_to_audio)["text"])
