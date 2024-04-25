## Cs7643 project


## Run 

## Save Dataset

`mmf_convert_hm --zip_file=~/cs7643-2/data.zip --password=1 --bypass_checksum=1`

`mmf_run config=projects/hateful_memes/configs/

## Run ViTEncBert
```sh
mmf_run config=projects/hateful_memes/configs/vilbert/vitencbert.yaml \
    model=vitencbert \
    dataset=hateful_memes \
    run_type=train_val
```

Code Changes
- Slight bugfix for wandb logger
- Added code for the hateful memes dataset
- Adjust preprocessing code 


## VITBERT
```sh
mmf_run config=projects/hateful_memes/configs/vilbert/vitbert.yaml \
    model=vitbert \
    dataset=hateful_memes \
    run_type=train_val 
```
- Copy over embeddings
- Alignment of attention layers
    - There are 12 layers each for text and image
What was done