python main_pretrain.py \
--distributed \
--num_workers 6 \
--model decomp_tiny \
--file decomp_tiny \
--reduction channel \
--input_size 512 \
--batch_size 8 \
--epochs 15 \
--warmup_epochs 0 \
--blr 1e-5 \
--data_path /Path/To/CTRATE \

