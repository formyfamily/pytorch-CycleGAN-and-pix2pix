python train_texture.py \
--name face_pix2pix_cond_geo2tex_128 \
--display_port 12334 \
--input_nc 6 \
--lambda_L1 100 \
--no_dropout \
--gpu_ids 4 \
--dataroot /home/ICT2000/jli/local/data \
--model pix2pixtex \
--no_flip \
--num_threads 0 \
--jsonfile ./train_all_data.json \
--display_freq 300 \
--print_freq 10
