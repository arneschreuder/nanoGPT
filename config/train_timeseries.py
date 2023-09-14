# train a miniature character-level shakespeare model
# good for debugging and playing on macbooks and such

out_dir = "out-timeseries"
eval_interval = 20  # keep frequent because we'll overfit
eval_iters = 100
log_interval = 10  # don't print too too often

# we expect to overfit on this small dataset, so only save when val improves
always_save_checkpoint = False

wandb_log = False  # override via command line if you like
wandb_project = "timeseries"
wandb_run_name = "gpt-timeseries"

dataset = "timeseries"
gradient_accumulation_steps = 1
batch_size = 32
block_size = 10  # context of up to 256 previous characters

# baby GPT model :)
n_layer = 1
n_head = 1
n_embd = 1
dropout = 0.2

learning_rate = 1e-3  # with baby networks can afford to go a bit higher
max_iters = 200
lr_decay_iters = 5000  # make equal to max_iters usually
min_lr = 1e-4  # learning_rate / 10 usually
beta2 = 0.99  # make a bit bigger because number of tokens per iter is small

warmup_iters = 100  # not super necessary potentially

# on macbook also add
device = "cpu"  # run on cpu only
compile = False  # do not torch compile the model
