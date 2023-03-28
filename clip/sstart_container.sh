# CUDA_VISIBLE_DEVICES=0 source launch_container.sh $PATH_TO_STORAGE/raw_video_dir $PATH_TO_STORAGE/feature_output_dir
CUDA_VISIBLE_DEVICES=1 source launch_container.sh ~/data/charades/ ~/HERO_Video_Feature_Extractor/output_test/
