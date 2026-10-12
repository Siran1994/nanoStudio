# train a miniature Chinese character-level model on 《雨夜谈经》
out_dir = 'checkpoints/llm_yuye_tanjing'
eval_interval = 100
eval_iters = 20
log_interval = 10

always_save_checkpoint = True

dataset = 'yuye_tanjing'
gradient_accumulation_steps = 1
batch_size = 16
block_size = 128

# Model architecture (Tiny ~4.7M params for Chinese)
n_layer = 6
n_head = 8
n_embd = 256
dropout = 0.1

learning_rate = 8e-4
max_iters = 2000
lr_decay_iters = 2000
min_lr = 8e-5
beta2 = 0.99

warmup_iters = 50
device = 'cpu'
compile = False
