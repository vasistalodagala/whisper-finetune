import argparse
from datasets import Dataset, Audio, Value

parser = argparse.ArgumentParser(description='Preliminary data preparation script before Whisper Fine-tuning.')
parser.add_argument('--source_data_dir', type=str, required=True, default=False, help='Path to the directory containing the audio_paths and text files.')
parser.add_argument('--output_data_dir', type=str, required=False, default='op_data_dir', help='Output data directory path.')

args = parser.parse_args()

scp_entries = open(f"{args.source_data_dir}/audio_paths", 'r').readlines()
txt_entries = open(f"{args.source_data_dir}/text", 'r').readlines()

if len(scp_entries) == len(txt_entries):
    audio_dataset = Dataset.from_dict({"audio": [audio_path.split()[1].strip() for audio_path in scp_entries],
                    "sentence": [' '.join(text_line.split()[1:]).strip() for text_line in txt_entries]})

    audio_dataset = audio_dataset.cast_column("audio", Audio(sampling_rate=16_000))
    audio_dataset = audio_dataset.cast_column("sentence", Value("string"))
    audio_dataset.save_to_disk(args.output_data_dir)
    print('Data preparation done')

else:
    print('Please re-check the audio_paths and text files. They seem to have a mismatch in terms of the number of entries. Both these files should be carrying the same number of lines.')
