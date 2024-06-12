gpu=0

CUDA_VISIBLE_DEVICES=$gpu python run.py \
    --gpu $gpu \
    --mode 'train' \
    --use_wandb 0 \
    --dim_emb 128 \
    --top_k 10 \
    --state_size 10 \
    --dim_actor 128 \
    --dim_critic 128 \
    --lr_actor 1e-4 \
    --lr_critic 1e-4 \
    --discount 0.9 \
    --batch_size 32 \
    --memory_size 1000000 \
    --tau 0.001 \
    --max_episode_num 10000 \
    --epsilon 1.0 \
    --std 1.5\
    --modality 'video,audio,text'\
    --fusion 'early' \
    --aggregation 'concat' \