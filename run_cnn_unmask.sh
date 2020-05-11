python main.py --glove [GLOVE PATH] \
               --dataset [TACRED JSON DATASET PAHT] \
               --output result/cnn_unmask \
               --lr 0.1 --batch_size 32 --weight_decay 1e-3
               --filter_num 500 --eval_batch_size 2048 \
               --optim adagrad --epoch 50 \
               --shuffle \
               --lr_dec_epoch 15 \
               --label_weights \
               --vocab 128000 \
			   --emb_freeze \
			   --emb_unfreeze 10 \
               --do_eval