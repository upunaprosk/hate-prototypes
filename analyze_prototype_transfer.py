#!/usr/bin/env python3

from model_finetuning import main


if __name__ == "__main__":
    main(
        default_model="facebook/opt-125m",
        decoder_only=True,
        default_seeds="5,6,7,8,9",
    )