import argparse
import os
from tts_model import load_chat_tts_model, tts
from config import DEFAULT_SPEED, DEFAULT_ORAL, DEFAULT_LAUGH, DEFAULT_BK, DEFAULT_SEG_LENGTH, DEFAULT_BATCH_SIZE

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate TTS audio from text file.")
    parser.add_argument("--text_file", type=str, required=True, help="Path to the text file to convert.")
    parser.add_argument("--seed", type=int,
                        help="Specific seed for generating audio. If not provided, seeds will be random.")
    parser.add_argument("--speed", type=int, default=DEFAULT_SPEED, help="Speed of generated audio.")
    parser.add_argument("--oral", type=int, default=DEFAULT_ORAL, help="Oral")
    parser.add_argument("--laugh", type=int, default=DEFAULT_LAUGH, help="Laugh")
    parser.add_argument("--bk", type=int, default=DEFAULT_BK, help="Break")
    parser.add_argument("--seg", type=int, default=DEFAULT_SEG_LENGTH, help="Max len of text segments.")
    parser.add_argument("--batch", type=int, default=DEFAULT_BATCH_SIZE, help="Batch size for TTS inference.")
    parser.add_argument("--source", type=str, default="huggingface", help="Model source: 'huggingface' or 'local'.")
    parser.add_argument("--local_path", type=str, help="Path to local model if source is 'local'.")

    args = parser.parse_args()
    chat = load_chat_tts_model(source=args.source, local_path=args.local_path)
    # chat = None
    tts(chat, args.text_file, args.seed, args.speed, args.oral, args.laugh, args.bk, args.seg,
        args.batch)
