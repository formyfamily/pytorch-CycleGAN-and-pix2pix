python train.py \
--name face_pix2pix_alpha_bs \
--display_port 10002 \
--input_nc 9 \
--lambda_L1 100 \
--no_dropout \
--objpath ./result_obj \
--gpu_ids 9 \
--dataroot /home/ICT2000/jli/local/data \
--model pix2pixbs \
--direction AtoB \
--load_size 256 \
--no_flip \
--dataset_mode facebs \
--num_threads 0 \
--jsonfile ./train_all_data.json
