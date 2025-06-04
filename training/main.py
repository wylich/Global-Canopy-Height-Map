
import getpass
import os
import shutil
import socket
import sys
import tempfile
import warnings
from contextlib import contextmanager

import wandb

from runner import Runner
from utilities import GeneralUtility

warnings.filterwarnings('ignore')

debug = "--debug" in sys.argv

defaults = dict(
    # System
    seed=1,

    # Data
    dataset='ai4forest_camera', # previously: ai4forest_debug
    batch_size=12, # prev. 5 (12 for colab on resnet50)

    # Architecture
    arch='unet',  # Defaults to unet
    backbone='resnet50',  # Defaults to resnet50
    use_pretrained_model=False,

    # Optimization
    optim='AdamW',  # Defaults to AdamW
    loss_name='shift_huber',  # Defaults to shift_l1
    n_iterations=1000, # batches processed
    log_freq=50, #default 5
    initial_lr=1e-3,
    weight_decay=1e-3, # 0.001 as in the paper
    use_standardization=True,
    use_augmentation=False, # can be set to true for image rotation
    use_label_rescaling=False,

    # Efficiency
    fp16=False,
    use_memmap=False,
    num_workers_per_gpu=12,   # Defaults to 8

    # Other
    use_weighted_sampler=False, # ='g10',
    use_weighting_quantile=None, # =10,
    use_swa=False,
    use_mixup=False,
    use_grad_clipping=True,
    use_input_clipping=False,   # Must be in [False, None, 1, 2, 5]
    n_lr_cycles=0,
    cyclic_mode='triangular2',
    )

if not debug:
    # Set everything to None recursively
    defaults = GeneralUtility.fill_dict_with_none(defaults)

# Add the hostname to the defaults
defaults['computer'] = socket.gethostname()

# Configure wandb logging
wandb.init(
    config=defaults,
    project='test-000',  # automatically changed in sweep
    entity=None,  # automatically changed in sweep
)
config = wandb.config
config = GeneralUtility.update_config_with_default(config, defaults)
# print(f"Using config: {config}")

@contextmanager
def tempdir():
    """Context manager for temporary directory that works on Windows and Linux."""
    username = getpass.getuser()
    
    # Windows-specific temp directory handling
    if os.name == 'nt':
        # Windows will always use default temp directory
        path = tempfile.mkdtemp()
    else:
        # Unix/Linux path handling (kept for compatibility)
        tmp_root = os.path.join('/scratch/local/', username)
        tmp_path = os.path.join(tmp_root, 'tmp')
        
        if os.path.isdir('/scratch/local/'):
            if not os.path.isdir(tmp_root):
                os.mkdir(tmp_root)
            if not os.path.isdir(tmp_path):
                os.mkdir(tmp_path)
            path = tempfile.mkdtemp(dir=tmp_path)
        else:
            path = tempfile.mkdtemp()
    
    try:
        yield path
    finally:
        try:
            shutil.rmtree(path)
            sys.stdout.write(f"Removed temporary directory {path}.\n")
        except IOError as e:
            sys.stderr.write(f'Failed to clean up temp dir {path}: {str(e)}\n')


with tempdir() as tmp_dir:
    # Platform-agnostic hostname detection
    hostname = None
    try:
        # Try Unix method first
        hostname = os.uname().nodename
    except AttributeError:
        # Fallback to Windows method
        hostname = os.environ.get('COMPUTERNAME', socket.gethostname())
    
    # Check environment conditions
    is_htc = hostname and 'htc-' in hostname.lower()
    is_gcp = hostname and 'gpu' in hostname.lower() and not is_htc
    
    if is_gcp:
        print('Running on GCP, marking as preemptable.')
        wandb.mark_preempting() # Note: This potentially overwrites the config when a run is resumed -> problems with tmp_dir

    runner = Runner(config=config, tmp_dir=tmp_dir, debug=debug)
    runner.run()

    # Close wandb run
    wandb_dir_path = wandb.run.dir
    wandb.join()

    # Delete the local files
    if os.path.exists(wandb_dir_path):
        try:
            shutil.rmtree(wandb_dir_path)
            print(f"Removed wandb directory {wandb_dir_path}")
        except Exception as e:
            print(f"Failed to remove wandb directory: {str(e)}")

