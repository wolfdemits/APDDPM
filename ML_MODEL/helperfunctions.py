import matplotlib.pyplot as plt

def export_metrics(metrics_dict, metric_type):
    # TODO
    if metric_type == 'train-batch':
        print(metrics_dict['batch_loss'])
    elif metric_type == 'train-epoch':
        print(metrics_dict['epoch_loss'])
    elif metric_type == 'val-batch':
        print(metrics_dict['batch_loss'])
    elif metric_type == 'val-epoch':
        print(metrics_dict['epoch_loss'])
    else:
        return