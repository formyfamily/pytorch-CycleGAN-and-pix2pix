python train.py \
--name face_pix2pix_alpha_bs_gan_debug \
--display_port 11112 \
--input_nc 9 \
--lambda_L1 100 \
--no_dropout \
--objpath ./result_obj_gan_debug \
--gpu_ids 5 \
--dataroot /home/ICT2000/jli/local/data \
--model pix2pixbs \
--direction AtoB \
--load_size 256 \
--no_flip \
--num_threads 0 \
--jsonfile ./train_all_data.json \
--alternate_epoch -1 \
--alternate_iter 5 \
--display_freq 10 \
--print_freq 1