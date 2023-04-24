import os
import argparse
import evaluate
from tqdm import tqdm
import jax.numpy as jnp
from datasets import Audio, load_from_disk
from transformers.models.whisper.english_normalizer import BasicTextNormalizer
from whisper_jax import FlaxWhisperForConditionalGeneration, FlaxWhisperPipline

wer_metric = evaluate.load("wer")
cer_metric = evaluate.load("cer")

def is_target_text_in_range(ref):
    if ref.strip() == "ignore time segment in scoring":
        return False
    else:
        return ref.strip() != ""


def get_text(sample):
    if "text" in sample:
        return sample["text"]
    elif "sentence" in sample:
        return sample["sentence"]
    elif "normalized_text" in sample:
        return sample["normalized_text"]
    elif "transcript" in sample:
        return sample["transcript"]
    elif "transcription" in sample:
        return sample["transcription"]
    else:
        raise ValueError(
            f"Expected transcript column of either 'text', 'sentence', 'normalized_text' or 'transcript'. Got sample of "
            ".join{sample.keys()}. Ensure a text column name is present in the dataset."
        )


def get_text_column_names(column_names):
    if "text" in column_names:
        return "text"
    elif "sentence" in column_names:
        return "sentence"
    elif "normalized_text" in column_names:
        return "normalized_text"
    elif "transcript" in column_names:
        return "transcript"
    elif "transcription" in column_names:
        return "transcription"


whisper_norm = BasicTextNormalizer()
def normalise(batch):
    batch["norm_text"] = whisper_norm(get_text(batch))
    return batch


def data(dataset):
    for i, item in enumerate(dataset):
        yield {**item["audio"], "reference": get_text(item), "norm_reference": item["norm_text"]}


def main(args):
    
    model_id = args.hf_model

    if args.half_precision == False:
        whisper_asr = FlaxWhisperPipline(
            model_id
        )
    else:
        whisper_asr = FlaxWhisperPipline(
            model_id,
            dtype=jnp.float16
        )

    whisper_asr.model.config.forced_decoder_ids = (
        whisper_asr.tokenizer.get_decoder_prompt_ids(
            language=args.language, task="transcribe"
        )
    )

    os.system(f"mkdir {args.output_dir}")
    for dset in args.eval_datasets:
        print('\nInfering on the dataset : ', dset)
        dataset = load_from_disk(dset)
        text_column_name = get_text_column_names(dataset.column_names)
        dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
        dataset = dataset.map(normalise, num_proc=2)
        dataset = dataset.filter(is_target_text_in_range, input_columns=[text_column_name], num_proc=2)

        predictions = []
        references = []
        norm_predictions = []
        norm_references = []
        
        for utt in tqdm(data(dataset), desc='Decode Progress'):
            out = whisper_asr(utt['array'])
            predictions.append(out["text"])
            references.append(utt["reference"])
            norm_predictions.append(whisper_norm(out["text"]))
            norm_references.append(utt["norm_reference"])

        wer = wer_metric.compute(references=references, predictions=predictions)
        wer = round(100 * wer, 2)
        cer = cer_metric.compute(references=references, predictions=predictions)
        cer = round(100 * cer, 2)
        norm_wer = wer_metric.compute(references=norm_references, predictions=norm_predictions)
        norm_wer = round(100 * norm_wer, 2)
        norm_cer = cer_metric.compute(references=norm_references, predictions=norm_predictions)
        norm_cer = round(100 * norm_cer, 2)

        print("WER : ", wer)
        print("CER : ", cer)
        print("\nNORMALIZED WER : ", norm_wer)
        print("NORMALIZED CER : ", norm_cer)

        dset = dset.replace('/', '_')
        op_file = args.output_dir + '/' + dset
        op_file = op_file + '_' + args.hf_model.replace('/', '_')
        result_file = open(op_file, 'w')
        result_file.write('\nWER: ' + str(wer) + '\n')
        result_file.write('CER: ' + str(cer) + '\n')
        result_file.write('\nNORMALIZED WER: ' + str(norm_wer) + '\n')
        result_file.write('NORMALIZED CER: ' + str(norm_cer) + '\n\n\n')

        for ref, hyp in zip(references, predictions):
            result_file.write('REF: ' + ref + '\n')
            result_file.write('HYP: ' + hyp + '\n')
            result_file.write("------------------------------------------------------" + '\n')
        result_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--hf_model",
        type=str,
        required=False,
        default="openai/whisper-tiny",
        help="Huggingface model name. Example: openai/whisper-tiny",
    )
    parser.add_argument(
        "--language",
        type=str,
        required=False,
        default="hi",
        help="Two letter language code for the transcription language, e.g. use 'hi' for Hindi. This helps initialize the tokenizer.",
    )
    parser.add_argument(
        "--eval_datasets", 
        type=str, 
        nargs='+', 
        required=True, 
        default=[], 
        help="List of datasets to evaluate the model on."
    )
    parser.add_argument(
        "--device",
        type=int,
        required=False,
        default=0,
        help="The device to run the pipeline on. -1 for CPU, 0 for the first GPU (default) and so on.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        required=False,
        default=16,
        help="Number of samples to go through each streamed batch.",
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        required=False, 
        default="jax_predictions_dir", 
        help="Output directory for the predictions and hypotheses generated."
    )
    parser.add_argument(
        "--half_precision",
        required=False,
        default=False, 
        type=lambda x: (str(x).lower() == 'true'),
        help="Run with half precision.",
    )

    args = parser.parse_args()
    main(args)
