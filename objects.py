import os
import time
import yaml

from ws.apis import *
from ws.shared.read_cfg import *
from ws.shared.logger import *
from train import train

from ws.shared.worker import WorkerResource

RESOURCE = WorkerResource() # allocated computing resource identifier
START_TIME = None

def get_resource():
    global RESOURCE
    return RESOURCE


def create_yaml_config(config, max_epoch):

    run_id = get_resource().get_id()
    
    try:
        with open('configs/cifar10.yaml') as template:
            conf_dict = yaml.load(template, Loader=yaml.FullLoader)
        
        # update configuration
        conf_dict['trainer']['num_epochs'] = max_epoch
        milestones = [int(0.3 * max_epoch), int(0.6 * max_epoch), int(0.8 * max_epoch)]
        conf_dict['scheduler']['milestones'] = milestones

        conf_dict['trainer']['output_dir'] = 'experiments/cifar10-{}'.format(run_id)
        
        if "batch_size" in config:
            batch_size = config['batch_size']
            conf_dict['dataset']['batch_size'] = batch_size

        if "lr" in config:
            lr = config['lr']
            conf_dict['optimizer']['lr'] = lr

        if 'optimizer' in config:
            opt = config['optimizer']
            conf_dict['optimizer']['name'] = opt

        if 'model_type' in config:
            m = config['model_type']
            conf_dict['model']['name'] = 'efficientnet_{}'.format(m)

    except Exception as ex:
        warn("Invalid configuration: {}".format(config))
    
    cfg_path = "configs/cifar10-{}.yaml".format(run_id)

    with open(cfg_path, 'w') as f:
        yaml.dump(conf_dict, f)

    return cfg_path


def update_epoch_acc(num_epoch, valid_acc):
    global START_TIME
    if valid_acc > 1.0:
        valid_acc = float(valid_acc / 100.0)
    debug("validation accuracy at {}: {}".format(num_epoch, valid_acc))
    valid_error = 1.0 - float(valid_acc)
    elapsed_time = time.time() - START_TIME
    update_current_loss(num_epoch, valid_error, elapsed_time)


''' SOTA classification problem '''
@objective_function
def tune_efficientnet_cifar10(config, fail_err=0.9, **kwargs):
    global START_TIME
    START_TIME = time.time()

    max_epoch = 90
    if "max_iters" in kwargs:
        if "iter_unit" in kwargs and kwargs["iter_unit"] == "epoch":
            max_epoch = kwargs["max_iters"]   

    cfg_path = create_yaml_config(config, max_epoch)

    train(cfg_path, epoch_cb=update_epoch_acc)


if __name__ == "__main__":
    # for config test only
    config = {'batch_size': 16, 'lr': 1.5e-3, 'optimizer': 'Adam', 'model_type': 'b1'}
    create_yaml_config(config, 10)