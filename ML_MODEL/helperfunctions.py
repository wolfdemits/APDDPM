import matplotlib.pyplot as plt
import numpy as np
import torch

loss_criterion = torch.nn.MSELoss()

def export_metrics(metrics_dict, type, FIGUREPATH, VMAX_PERCENTILE=99):
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
        xt = metrics_dict['xt']
        xT = metrics_dict['xT']

        patients = batch['Patient']
        planes = batch['Plane']
        slice_idxs = batch['SliceIndex']
        flip = metrics_dict['flip']

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
            if patient in view_patients:
                idx = slice_idxs[i]

                if int(idx) in list(slices):
                    # loss
                    with torch.no_grad():
                        loss = loss_criterion(x_t_1[i], x_t_1_hat[i])
                        loss = loss.item()

                    img1 = xt[i].squeeze().cpu()
                    img2 = x_t_1_hat[i].squeeze().cpu()
                    img3 = x_t_1[i].squeeze().cpu()
                    img4 = x0_hat[i].squeeze().cpu()
                    img5 = x0[i].squeeze().cpu()
                    img6 = xT[i].squeeze().cpu()

                    # undo flipping
                    if flip[i][0]:
                        img1 = torch.flip(img1, dims=[0])
                        img2 = torch.flip(img2, dims=[0])
                        img3 = torch.flip(img3, dims=[0])
                        img4 = torch.flip(img4, dims=[0])
                        img5 = torch.flip(img5, dims=[0])
                        img6 = torch.flip(img6, dims=[0])
                    if flip[i][1]:
                        img1 = torch.flip(img1, dims=[1])
                        img2 = torch.flip(img2, dims=[1])
                        img3 = torch.flip(img3, dims=[1])
                        img4 = torch.flip(img4, dims=[1])
                        img5 = torch.flip(img5, dims=[1])
                        img6 = torch.flip(img6, dims=[1])

                    vmax = np.percentile(np.array([img1.detach(),img2.detach(),img3.detach(),img4.detach(),img5.detach(),img6.detach()]), VMAX_PERCENTILE)

                    fig, ax = plt.subplots(2, 3, figsize=(11, 7))
                    ax[0,0].imshow(img1, cmap='gray_r', vmin=0, vmax=vmax)
                    ax[0,0].set_title('IN: xt')
                    ax[0,1].imshow(img2.detach(), cmap='gray_r', vmin=0, vmax=vmax)
                    ax[0,1].set_title('OUT: x^_t-1')
                    ax[0,2].imshow(img3.detach(), cmap='gray_r', vmin=0, vmax=vmax)
                    ax[0,2].set_title('TARGET: x_t-1')
                    ax[1,0].imshow(img4.detach(), cmap='gray_r', vmin=0, vmax=vmax)
                    ax[1,0].set_title('UNet: x0^')
                    ax[1,1].imshow(img5.detach(), cmap='gray_r', vmin=0, vmax=vmax)
                    ax[1,1].set_title('HD: x0')
                    ax[1,2].imshow(img6.detach(), cmap='gray_r', vmin=0, vmax=vmax)
                    ax[1,2].set_title('LD: xT')
                    

                    path = FIGUREPATH / 'TRAIN' / f'epoch{epoch}' / f'{planes[i]}' / f'{patient}'
                    path.mkdir(exist_ok=True, parents=True)
                    fig.suptitle(f'Example train images: epoch {epoch}, batch {current_batch}, patient {patient}, slice: {idx}, \n timestep fraction: {alpha[i]}, dose delta: {delta[i]}, loss: {loss:.4f}')
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
        xt = metrics_dict['xt']
        xT = metrics_dict['xT']

        patients = batch['Patient']
        planes = batch['Plane']
        slice_idxs = batch['SliceIndex']
        flip = metrics_dict['flip']

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
            if patient in view_patients:
                idx = slice_idxs[i]

                if int(idx) in list(slices):
                    # loss
                    with torch.no_grad():
                        loss = loss_criterion(x_t_1[i], x_t_1_hat[i])
                        loss = loss.item()

                    img1 = xt[i].squeeze().cpu()
                    img2 = x_t_1_hat[i].squeeze().cpu()
                    img3 = x_t_1[i].squeeze().cpu()
                    img4 = x0_hat[i].squeeze().cpu()
                    img5 = x0[i].squeeze().cpu()
                    img6 = xT[i].squeeze().cpu()

                    # undo flipping
                    if flip[i][0]:
                        img1 = torch.flip(img1, dims=[0])
                        img2 = torch.flip(img2, dims=[0])
                        img3 = torch.flip(img3, dims=[0])
                        img4 = torch.flip(img4, dims=[0])
                        img5 = torch.flip(img5, dims=[0])
                        img6 = torch.flip(img6, dims=[0])
                    if flip[i][1]:
                        img1 = torch.flip(img1, dims=[1])
                        img2 = torch.flip(img2, dims=[1])
                        img3 = torch.flip(img3, dims=[1])
                        img4 = torch.flip(img4, dims=[1])
                        img5 = torch.flip(img5, dims=[1])
                        img6 = torch.flip(img6, dims=[1])

                    vmax = np.percentile(np.array([img1.detach(),img2.detach(),img3.detach(),img4.detach(),img5.detach(),img6.detach()]), VMAX_PERCENTILE)

                    fig, ax = plt.subplots(2, 3, figsize=(11, 7))
                    ax[0,0].imshow(img1.detach(), cmap='gray_r', vmin=0, vmax=vmax)
                    ax[0,0].set_title('IN: xt')
                    ax[0,1].imshow(img2.detach(), cmap='gray_r', vmin=0, vmax=vmax)
                    ax[0,1].set_title('OUT: x^_t-1')
                    ax[0,2].imshow(img3.detach(), cmap='gray_r', vmin=0, vmax=vmax)
                    ax[0,2].set_title('TARGET: x_t-1')
                    ax[1,0].imshow(img4.detach(), cmap='gray_r', vmin=0, vmax=vmax)
                    ax[1,0].set_title('UNet: x0^')
                    ax[1,1].imshow(img5.detach(), cmap='gray_r', vmin=0, vmax=vmax)
                    ax[1,1].set_title('HD: x0')
                    ax[1,2].imshow(img6.detach(), cmap='gray_r', vmin=0, vmax=vmax)
                    ax[1,2].set_title('LD: xT')

                    path = FIGUREPATH / 'VAL' / f'epoch{epoch}' / f'{planes[i]}' / f'{patient}'
                    path.mkdir(exist_ok=True, parents=True)
                    fig.suptitle(f'Example val images: epoch {epoch}, batch {current_batch}, patient {patient}, slice: {idx}, \n timestep fraction: {alpha[i]}, dose delta: {delta[i]}, loss: {loss:.4f}')
                    fig.savefig(str(path / f'{idx}.png'))
                    plt.close()

    elif type == 'epoch-done':
        epoch_val_loss = metrics_dict['epoch_val_loss']
        epoch_train_loss = metrics_dict['epoch_train_loss']

        # plot loss
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        ax.plot(np.arange(len(epoch_train_loss)), epoch_train_loss, color='orange')
        ax.plot(np.arange(len(epoch_val_loss)), epoch_val_loss, color='blue')
        ax.legend(['train loss', 'val loss'])
        ax.set_title('Train/val loss')
        path = FIGUREPATH
        path.mkdir(exist_ok=True, parents=True)
        fig.savefig(str(path/ f'loss.png'))
        plt.close()

    else:
        return
    return