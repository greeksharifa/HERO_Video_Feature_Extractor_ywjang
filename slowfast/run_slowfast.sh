# rm -rf /output/slowfast_features/
python extract_feature/gather_video_paths.py --features_length $1
python extract_feature/extract_fixed_length.py --dataflow --csv /output/csv/slowfast_info_$1.csv \
    --batch_size 16 --num_decoding_thread 2 --clip_len 1\
    --features_length $1\
    TEST.CHECKPOINT_FILE_PATH /models/SLOWFAST_8x8_R50.pkl
