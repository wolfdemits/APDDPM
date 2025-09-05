import matplotlib.pyplot as plt
import numpy as np

def export_metrics(metrics_dict, metric_dict, FIGUREPATH):
    if metric_dict == 'train-batch':
        # plot loss
        batch_loss = metrics_dict['batch_loss']
        current_batch = metrics_dict['current_batch']
        slices_amount = metrics_dict['view_slices_amount']
        view_patients = metrics_dict['view_patients']
        batch = metrics_dict['batch']

        x_t_1 = metrics_dict['xt-1']
        x_t_1_hat = metrics_dict['xt-1_hat']
        x0 = metrics_dict['x0']
        x0_hat = metrics_dict['x0_hat']

        patients = batch['Patient']
        planes = batch['Plane']
        slice_idxs = batch['SliceIndex']

        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        ax.plot(np.arange(len(batch_loss)), batch_loss, color='blue')
        ax.set_title('Inter-epoch train loss')
        fig.savefig(str(FIGUREPATH / f'train_loss_batch'))
        plt.close()

        # TODO fix this mess
        for i, patient in enumerate(patients):
            if patient in view_patients:
                idx = slice_idxs[i]
                if int(idx) % 20 == 0:
                    # TODO: vmax
                    img1 = x_t_1[i].squeeze().cpu()
                    img2 = x_t_1_hat[i].squeeze().cpu()
                    img3 = x0[i].squeeze().cpu()
                    img4 = x0_hat[i].squeeze().cpu()
                    fig, ax = plt.subplots(2, 2, figsize=(5, 5))
                    ax[0,0].imshow(img1.detach(), cmap='gray_r')
                    ax[0,1].imshow(img2.detach(), cmap='gray_r')
                    ax[1,0].imshow(img3.detach(), cmap='gray_r')
                    ax[1,1].imshow(img4.detach(), cmap='gray_r')

                    fig.suptitle(f'Example images: batch {current_batch}, patient {patient}, slice: {idx}')
                    fig.savefig(str(FIGUREPATH / f'examples_{current_batch}_{patient}_{idx}'))
                    plt.close()

    elif metric_dict == 'train-epoch':
        print(metrics_dict['epoch_loss'])

    elif metric_dict == 'val-batch':
        # plot loss
        batch_loss = metrics_dict['batch_loss']
        current_batch = metrics_dict['current_batch']

        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        ax.plot(np.arange(len(batch_loss)), batch_loss, color='orange')
        ax.set_title('Inter-epoch val loss')
        fig.savefig(str(FIGUREPATH / f'val_loss_batch'))
        plt.close()

    elif metric_dict == 'val-epoch':
        print(metrics_dict['epoch_loss'])

    else:
        return
    return