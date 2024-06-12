gpu=0

CUDA_VISIBLE_DEVICES=$gpu python run.py \
    --gpu $gpu \
    --mode 'train' \
    --use_wandb 0 \
    --dim_emb 100 \
    --top_k 10 \
    --state_size 10 \
    --dim_actor 128 \
    --dim_critic 128 \
    --lr_actor 1e-4 \
    --lr_critic 1e-4 \
    --discount 0.9 \
    --batch_size 64 \
    --memory_size 1000000 \
    --tau 0.001 \
    --max_episode_num 50000 \
    --epsilon 1.0 \
    --std 1.5\
    --modality ''\
    