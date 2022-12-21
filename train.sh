for (( PART=1; PART<8; PART++))
do
    CUDA_VISIBLE_DEVICES=6 python train.py --config config_nn$PART.json --checkpoint_path ckpt_nn$PART
done