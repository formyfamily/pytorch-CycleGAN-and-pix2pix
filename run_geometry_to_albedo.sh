CUDA_VISIBLE_DEVICES=5 python train.py \
--objpath ./result_obj/face_geo2alb \
--gpu_ids 0 \
--dataroot /home/ICT2000/jli/local/data \
--name face_geo2alb \
--model pix2pix \
--direction AtoB \
--load_size 256 \
--no_flip \
--dataset_mode facex \
--num_threads 4