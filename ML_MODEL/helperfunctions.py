import matplotlib.pyplot as plt
import numpy as np
import torch

def export_metrics(metrics_dict, type, FIGUREPATH, VMAX_FRAC=0.5):
    if type == 'train-batch':
        # plot loss
        batch_loss = metrics_dict['batch_loss']
        current_batch = metrics_dict['current_batch']
        slices = metrics_dict['view_slices']
        view_patients = metrics_dict['view_patients']
        batch = metrics_dict['batch']
        epoch = metrics_dict['epoch']
        alpha = metrics_dict['pathfrac']
        delta = metrics_dict['delta']

        x_t_1 = metrics_dict['xt-1']
        x_t_1_hat = metrics_dict['xt-1_hat']
        x0 = metrics_dict['x0']
        x0_hat = metrics_dict['x0_hat']

        patients = batch['Patient']
        planes = batch['Plane']
        slice_idxs = batch['SliceIndex']

        # loss plots
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        ax.plot(np.arange(len(batch_loss)), batch_loss, color='orange')
        ax.set_title('Inter-epoch train loss')
        path = FIGUREPATH / 'TRAIN' / f'epoch{epoch}'
        path.mkdir(exist_ok=True, parents=True)
        fig.savefig(str(path  / f'train_loss_epoch.png'))
        plt.close()

        # plot slices
        for i, patient in enumerate(patients):
            for plane in planes:
                if patient in view_patients:
                    idx = slice_idxs[i]

                    if int(idx) in list(slices):
                        img1 = x_t_1[i].squeeze().cpu()
                        img2 = x_t_1_hat[i].squeeze().cpu()
                        img3 = x0[i].squeeze().cpu()
                        img4 = x0_hat[i].squeeze().cpu()

                        vmax = max(torch.max(img1), torch.max(img2), torch.max(img3), torch.max(img4))

                        fig, ax = plt.subplots(2, 2, figsize=(10, 7))
                        ax[0,0].imshow(img1.detach(), cmap='gray_r', vmin=0, vmax=vmax*VMAX_FRAC)
                        ax[0,0].set_title('x_t-1')
                        ax[0,1].imshow(img2.detach(), cmap='gray_r', vmin=0, vmax=vmax*VMAX_FRAC)
                        ax[0,1].set_title('x^_t-1')
                        ax[1,0].imshow(img3.detach(), cmap='gray_r', vmin=0, vmax=vmax*VMAX_FRAC)
                        ax[1,0].set_title('x0')
                        ax[1,1].imshow(img4.detach(), cmap='gray_r', vmin=0, vmax=vmax*VMAX_FRAC)
                        ax[1,1].set_title('x0^')

                        path = FIGUREPATH / 'TRAIN' / f'epoch{epoch}' / f'{plane}' / f'{patient}'
                        path.mkdir(exist_ok=True, parents=True)
                        fig.suptitle(f'Example train images: epoch {epoch}, batch {current_batch}, patient {patient}, slice: {idx}, \n timestep fraction: {alpha[i]}, dose delta: {delta[i]}')
                        fig.savefig(str(path  / f'{idx}.png'))
                        plt.close()

    elif type == 'val-batch':
        batch_loss = metrics_dict['batch_loss']
        current_batch = metrics_dict['current_batch']
        slices = metrics_dict['view_slices']
        view_patients = metrics_dict['view_patients']
        batch = metrics_dict['batch']
        epoch = metrics_dict['epoch']
        alpha = metrics_dict['pathfrac']
        delta = metrics_dict['delta']

        x_t_1 = metrics_dict['xt-1']
        x_t_1_hat = metrics_dict['xt-1_hat']
        x0 = metrics_dict['x0']
        x0_hat = metrics_dict['x0_hat']

        patients = batch['Patient']
        planes = batch['Plane']
        slice_idxs = batch['SliceIndex']

        # loss plot
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        ax.plot(np.arange(len(batch_loss)), batch_loss, color='blue')
        ax.set_title('Inter-epoch val loss')
        path = FIGUREPATH / 'VAL' / f'epoch{epoch}'
        path.mkdir(exist_ok=True, parents=True)
        fig.savefig(str(path / f'val_loss_epoch.png'))
        plt.close()

        # plot slices
        for i, patient in enumerate(patients):
            for plane in planes:
                if patient in view_patients:
                    idx = slice_idxs[i]

                    if int(idx) in list(slices):
                        img1 = x_t_1[i].squeeze().cpu()
                        img2 = x_t_1_hat[i].squeeze().cpu()
                        img3 = x0[i].squeeze().cpu()
                        img4 = x0_hat[i].squeeze().cpu()

                        vmax = max(torch.max(img1), torch.max(img2), torch.max(img3), torch.max(img4))

                        fig, ax = plt.subplots(2, 2, figsize=(5, 5))
                        ax[0,0].imshow(img1.detach(), cmap='gray_r', vmin=0, vmax=vmax*VMAX_FRAC)
                        ax[0,0].set_title('x_t-1')
                        ax[0,1].imshow(img2.detach(), cmap='gray_r', vmin=0, vmax=vmax*VMAX_FRAC)
                        ax[0,1].set_title('x^_t-1')
                        ax[1,0].imshow(img3.detach(), cmap='gray_r', vmin=0, vmax=vmax*VMAX_FRAC)
                        ax[1,0].set_title('x0')
                        ax[1,1].imshow(img4.detach(), cmap='gray_r', vmin=0, vmax=vmax*VMAX_FRAC)
                        ax[1,1].set_title('x0^')

                        path = FIGUREPATH / 'VAL' / f'epoch{epoch}' / f'{plane}' / f'{patient}'
                        path.mkdir(exist_ok=True, parents=True)
                        fig.suptitle(f'Example val images: epoch {epoch}, batch {current_batch}, patient {patient}, slice: {idx}, \n timestep fraction: {alpha[i]}, dose delta: {delta[i]}')
                        fig.savefig(str(path / f'{idx}.png'))
                        plt.close()

    elif type == 'epoch-done':
        epoch_val_loss = metrics_dict['epoch_val_loss']
        epoch_train_loss = metrics_dict['epoch_train_loss']

        # plot loss
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        ax.plot(np.arange(len(epoch_train_loss)), epoch_train_loss, color='orange')
        ax.plot(np.arange(len(epoch_val_loss)), epoch_val_loss, color='blue')
        ax.set_title('Train/val loss')
        path = FIGUREPATH
        path.mkdir(exist_ok=True, parents=True)
        fig.savefig(str(path/ f'loss.png'))
        plt.close()

    else:
        return
    return