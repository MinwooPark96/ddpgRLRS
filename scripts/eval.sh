gpu=-1
save_dir=./save_model/trail-2024-06-11-21-30-15

CUDA_VISIBLE_DEVICES=$gpu python run.py \
    --gpu $gpu \
    --mode 'eval' \
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
    --saved_actor $save_dir/actor_8000_fixed.pth \
    --saved_critic $save_dir/critic_8000_fixed.pth
    