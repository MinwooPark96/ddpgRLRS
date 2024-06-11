import argparse
import os
import torch
import logging
import random

import train
import eval

logging.basicConfig(format='%(levelname)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO,
                    # filename="./log/train_cross.log",
                    # filemode='w')
)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--gpu', '-g', type = int, help="gpu", default='-1') 
    parser.add_argument('--mode', '-m', type = str, help="mode", default='train')
    parser.add_argument('--checkpoint', '-ch', type = str, help ='checkpoint', default = "")
    parser.add_argument ('--use_wandb', '-w', help ='use wandb', type=int, default = 0)
    
    parser.add_argument('--dim_emb', '-e', type = int, help ='dimension of embedding', default=100)
    parser.add_argument('--top_k', '-k', type = int, help ='how many recommend items', default=5)  
    parser.add_argument('--state_size', '-s', type = int,help ='state size', default=10)  
    
    parser.add_argument('--dim_actor', '-ha', type = int, help ='dimension of hidden layer for actor network', default=128)
    parser.add_argument('--lr_actor', '-la',  type = float, help ='learning rate of hidden layer for actor network', default=0.001)
    
    parser.add_argument('--dim_critic', '-hc', type = int, help ='dimension of hidden layer for critic network', default=128)
    parser.add_argument('--lr_critic', '-lc', type = float, help ='learning rate of hidden layer for critic network', default=0.001)
    
    parser.add_argument('--discount', '-d', type = float, help ='discount factor', default=0.9)
    parser.add_argument('--batch_size', '-b', type = int, help ='batch size', default=32)
    parser.add_argument('--memory_size', '-ms', type = int, help ='memory size', default=1000000)
    parser.add_argument('--tau', '-t', help ='tau',type = float, default=0.001)
    parser.add_argument('--max_episode_num', '-me', type=int, help ='max episode number', default=50000)
    
    parser.add_argument('--epsilon', '-ep', type = float, help ='epsilon', default=1.0)
    parser.add_argument('--std', '-std', type = float, help ='standard deriviation of normal term', default=1.5)
    
    parser.add_argument('--modality', type=str, help='Modality', default='video,audio')
    parser.add_argument('--fusion', type=str, help='Fusion', default='')
    parser.add_argument('--aggregation', type=str, help='Aggregation', default='')
    
    parser.add_argument('--saved_actor',type = str, help ='directory of saved actor model')
    parser.add_argument('--saved_critic',type = str, help ='directory of saved actor model')
    
    os.system("clear")
    
    for arg in vars(parser.parse_args()).items():
        logger.info(arg)

    args = parser.parse_args()
    
    if args.mode == 'train':
        train.trainer(args = args)
    elif args.mode == 'eval':
        eval.evaluater(args = args)
    else :
        raise ValueError("Invalid mode")
    
    
    
