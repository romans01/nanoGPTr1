# train a miniature character-level shakespeare model
# good for debugging and playing on macbooks and such

out_dir = 'out-poems-char'
eval_interval = 200 # keep frequent because we'll overfit
eval_iters = 200
log_interval = 10 # don't print too too often

# we expect to overfit on this small dataset, so only save when val improves
always_save_checkpoint = False

wandb_log = True # override via command line if you like
wandb_project = 'russian-poems-char'
wandb_run_name = 'mini-gpt'

dataset = 'poems'
gradient_accumulation_steps = 1
batch_size = 256
block_size = 512 # context of up to 256 previous characters

# baby GPT model :)
n_layer = 8
n_head = 8
n_embd = 256
dropout = 0.1

wandb_run_name = 'mini-gpt-'+str(n_layer)+'-'+str(n_head)+"-"+str(n_embd)+"-"+str(block_size)

learning_rate = 7e-3 # with baby networks can afford to go a bit higher
max_iters = 10000
lr_decay_iters = 2500 # make equal to max_iters usually
min_lr = 1e-5 # learning_rate / 10 usually
beta2 = 0.9 # make a bit bigger because number of tokens per iter is small

warmup_iters = 100 # not super necessary potentially

# on macbook also add
# device = 'cpu'  # run on cpu only
# compile = False # do not torch compile the model
