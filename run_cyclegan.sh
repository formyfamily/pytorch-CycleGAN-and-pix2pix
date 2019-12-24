python train.py \
--objpath ./result_obj/face_cyclegan_bs \
--dataroot /home/ICT2000/jli/local/data \
--name face_cyclegan_bs \
--model cycle_gan \
--load_size 256 \
--no_flip \
--dataset_mode facex \
--num_threads 4
