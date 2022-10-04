# init train

```bash
python train_pt.py --model GC_VIT --batchsize 64 --normalize_head mean_std
```

WFNet
GC_VIT
GC_VIT_XXTINY

# tensorboard

```bash
tensorboard --logdir . --bind_all
```


# resume train

```bash
python train.py --resume --batchsize 64 --normalize_head mean_std --normalize_tail mean_std
```

# test

```bash
python do_test_n_write_mat.py --model GC_VIT --path_model GC_VIT-v=00-epoch=149-val_loss=2.39.ckpt --normalize_head mean_std && \
python do_test_n_write_mat.py --model GC_VIT_XXTINY --path_model GC_VIT_XXTINY-v=02-epoch=143-val_loss=2.34.ckpt --normalize_head mean_std && \
python do_test_n_write_mat.py --model WFNet --path_model WFNet-v=01-epoch=136-val_loss=53.60.ckpt --normalize_head mean_std && \
python test_metrics.py --model GC_VIT --path_model GC_VIT-v=00-epoch=149-val_loss=2.39.ckpt --normalize_head mean_std && \
python test_metrics.py --model GC_VIT_XXTINY --path_model GC_VIT_XXTINY-v=02-epoch=143-val_loss=2.34.ckpt --normalize_head mean_std && \
python test_metrics.py --model WFNet --path_model WFNet-v=01-epoch=136-val_loss=53.60.ckpt --normalize_head mean_std
```
