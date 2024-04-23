## Cs7643 project


## Run 

## Save Dataset

`mmf_convert_hm --zip_file=~/cs7643-2/data.zip --password=1 --bypass_checksum=1`

`mmf_run config=projects/hateful_memes/configs/

mmf_run config=projects/hateful_memes/configs/vilbert/vitbert.yaml \
    model=vitbert \
    dataset=hateful_memes \
    run_type=train_val