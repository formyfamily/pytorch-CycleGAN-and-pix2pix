CUDA_VISIBLE_DEVICES=5 python train.py \
--objpath ./result_obj/face_alb2geo \
--gpu_ids 0 \
--dataroot /home/ICT2000/jli/local/data \
--name face_alb2geo \
--model pix2pix \
--direction AtoB \
--load_size 256 \
--no_flip \
--dataset_mode facex \
--num_threads 4