CUDA_VISIBLE_DEVICES=1 python test_texture.py \
--name face_pix2pix_cond_geo2tex \
--input_nc 6 \
--no_dropout \
--gpu_ids 0 \
--dataroot /home/ICT2000/jli/local/data \
--model pix2pixtex \
--jsonfile ./train_all_data.json 
