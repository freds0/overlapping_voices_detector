import argparse
import collections
import torch
import numpy as np
import data_loader as module_data
from trainer import loss as module_loss
import model as module_arch
from utils import prepare_device
from utils.logger import logger
from utils.config_parser import ConfigParser
from utils.metric_handler import MetricHandler

from utils.charts import plot_confusion_matrix, plot_roc_curve
# fix random seeds for reproducibility
SEED = 42
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)


def resume_checkpoint(model, resume_path, logger):
    """
    Resume from saved checkpoints
    :param resume_path: Checkpoint path to be resumed
    """
    resume_path = str(resume_path)
    logger.info("Loading checkpoint: {} ...".format(resume_path))
    checkpoint = torch.load(resume_path)
    model.load_state_dict(checkpoint['state_dict'])
    logger.info("Checkpoint loaded.")

def format_metrics(metrics):
    """
    Function to dynamically create the metrics string
    """
    metric_str = ""
    for key, value in metrics.items():
        metric_str += "{}: {:.4f} ".format(key.capitalize(), value)
    return metric_str

def main(config):
    # setup data_loader instances
    data_loader = config.init_obj('data_loader', module_data)
    test_data_loader = data_loader.get_val_dataloader(set='test')

    # build model architecture, then print to console
    model = config.init_obj('arch', module_arch)
    resume_checkpoint(model, config.resume, logger)

    # get function handles of loss and metrics
    metrics_handler = MetricHandler(config['metrics'])
    criterion = getattr(module_loss, config['loss'])

    # prepare for (multi-device) GPU training
    device, device_ids = prepare_device(config['n_gpu'], logger)
    model = model.to(device)
    model.eval()
    metrics_handler.reset()
    outputs = torch.tensor([]).to(device)
    targets = torch.tensor([]).to(device)
    total_loss = 0
    with torch.no_grad():
        for batch_idx, input_data in enumerate(test_data_loader):
            data, target = input_data
            data, target = data.to(device), target.to(device)
            output = model(data).squeeze()
            output = output.reshape(-1)
            loss = criterion(output, target)
            total_loss += loss.item()
            prediction = output > 0.5
            outputs = torch.cat((outputs, prediction), dim=0)
            targets = torch.cat((targets, target), dim=0)


    metrics_handler.update(outputs, targets)
    metrics = metrics_handler.get_data()
    metrics_str = format_metrics(metrics)
    logger.info("Test: Loss: {:.4f} {}".format(total_loss, metrics_str))
    plot_confusion_matrix(outputs.to('cpu').detach().numpy(), targets.to('cpu').detach().numpy(), filename='confusion_matrix.png')

    plot_roc_curve(outputs.to('cpu').detach().numpy(), targets.to('cpu').detach().numpy(), filename='roc_curve.png')


if __name__ == '__main__':

    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', required=True, default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    config = ConfigParser.from_args(args, log_to_file=False)
    logger.config(folder=None)

    main(config)
