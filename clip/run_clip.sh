# rm -rf /output/clip-vit_features_64/
python gather_video_paths.py --features_length $1
python extract.py --csv /output/csv/clip-vit_info_$1.csv\
       	--num_decoding_thread 2 --model_version ViT-B/32\
	--features_length $1

