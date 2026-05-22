import yaml 
import os


config = os.environ.get("CONFIG")

with open(config, "r") as f:
    config_to_gpus = yaml.safe_load(f)

gpus = config_to_gpus['num_gpu']

torchrun_cmd = f'torchrun --nproc_per_node={gpus} --master_port=4321 basicsr/train.py -opt {config} --launcher pytorch'

print(torchrun_cmd)
os.system(torchrun_cmd)
