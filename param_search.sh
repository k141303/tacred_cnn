# CNN unmask
python param_search.py --glove /home/nkota/Python/RelationExtraction/TACRED/CNN/data/word2vec/glove.840B.300d.txt \
                       --dataset /home/nkota/DATASET/tacred/data/json \
                       --filter_num 500 --eval_batch_size 2048 \
                       --optim adagrad --epoch 50 \
                       --shuffle \
                       --lr_dec_epoch 15 \
                       --label_weights \
                       --vocab 128000 \
         			   --emb_freeze \
        			   --emb_unfreeze 10

# CNN mask
python param_search.py --glove /home/nkota/Python/RelationExtraction/TACRED/CNN/data/word2vec/glove.840B.300d.txt \
                       --dataset /home/nkota/DATASET/tacred/data/json \
                       --filter_num 500 --eval_batch_size 2048 \
                       --optim adagrad --epoch 50 \
                       --shuffle \
                       --lr_dec_epoch 15 \
                       --label_weights \
                       --vocab 128000 \
        			   --emb_freeze \
       			       --emb_unfreeze 10 \
                       --entity_mask
