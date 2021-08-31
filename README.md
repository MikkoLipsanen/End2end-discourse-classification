# End2end-discourse-classification

### Example parameters for running the LSTM model:

```
python lstm.py --batch_size 100 --lr 0.2 --epochs 10 --lstm_dropout 0.2 --lstm_layers 2 --hidden_size 10
```

### Example parameters for running the attention model:

```
python attention.py --batch_size 100 --epochs 10 --attn_dropout 0.2 --n_heads 2 --attn_layers 2
```
