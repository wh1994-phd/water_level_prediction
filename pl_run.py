import os
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
import argparse
import random
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
from exp.exp_basic import Exp_Basic
from data_provider.data_factory import data_provider
from models import DLGNet

class TimeSeriesLightningModule(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.save_hyperparameters()
        

        self.model_dict = {
            'DLGNet': DLGNet
        }
        

        self.model = self._build_model()
        
    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()
        return model
    
    def forward(self, batch_x, batch_x_mark, dec_inp, batch_y_mark):

        if self.args.output_attention:
            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
        else:
            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        return outputs
        
    def training_step(self, batch, batch_idx):
        batch_x, batch_y, batch_x_mark, batch_y_mark = batch
        batch_x = batch_x.float()
        batch_y = batch_y.float()
        batch_x_mark = batch_x_mark.float()
        batch_y_mark = batch_y_mark.float()
        

        if self.current_epoch == 0 and batch_idx == 0:
            print(f"batch_x shape: {batch_x.shape}")
            print(f"batch_y shape: {batch_y.shape}")
            print(f"batch_x_mark shape: {batch_x_mark.shape}")
            print(f"batch_y_mark shape: {batch_y_mark.shape}")
        

        dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
        dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float()
        outputs = self(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        

        if self.current_epoch == 0 and batch_idx == 0:
            print(f"outputs shape: {outputs.shape}")
        

        f_dim = -1 if self.args.features == 'MS' else 0
        outputs = outputs[:, -self.args.pred_len:, f_dim:]
        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
        loss = torch.nn.MSELoss()(outputs, batch_y)
        
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        

        if self.current_epoch % 20 == 0 and batch_idx == 0:
            self.visualize_predictions(batch_x, batch_y, outputs)
        
        return loss
    
    def visualize_predictions(self, batch_x, batch_y, outputs):

        plt.figure(figsize=(12, 6))
        full_true = torch.cat((batch_x[0, :, -1], batch_y[0, :, -1]), dim=0).cpu().numpy()
        plt.plot(range(len(full_true)), full_true, color='black', label='True Data')
        plt.plot(range(self.args.seq_len, self.args.seq_len + self.args.pred_len), outputs[0, :, 0].detach().cpu().numpy(), color='orange', label='Prediction')
        plt.legend()
        plt.title(f'Epoch {self.current_epoch} Predictions')
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.grid(True)
        plt.show()
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.args.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            factor=0.1,
            patience=self.args.patience,
            verbose=True
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss'
            }
        }


def main():
    parser = argparse.ArgumentParser(description='han chuan water level prediction')

    # basic config
    parser.add_argument('--task_name', type=str, default='long_term_forecast',
                        help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
    parser.add_argument('--is_training', type=int,  default=1, help='status')
    parser.add_argument('--model_id', type=str,  default='test', help='model id')
    parser.add_argument('--model', type=str,  default='DLGNet',
                        help='model name, options: [Autoformer, Transformer, DLGNet]')

    # data loader
    parser.add_argument('--data', type=str,  default='hj', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./data', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='hj.xlsx', help='data file')
    parser.add_argument('--features', type=str, default='MS',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='hj', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./new_check/', help='location of model checkpoints')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
    parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')
    """
    self.seq_len, self.wave_level, self.wave_type
    """
    parser.add_argument('--wave_level', type=int, default=6)
    parser.add_argument('--wave_type', type=str, default="db10")

    # model define
    parser.add_argument('--top_k', type=int, default=5, help='for DLGBlock')
    parser.add_argument('--num_kernels', type=int, default=6, help='for Inception')
    parser.add_argument('--enc_in', type=int, default=8, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=8, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=8, help='output size')
    parser.add_argument('--d_model', type=int, default=16, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')

    # DLGNet
    parser.add_argument('--muti_kenerl', type=int, default=3, help='kenerl number in mti-scale conv')
    parser.add_argument('--level', type=int, default=2,
                        help='deepth of causal conv')
    parser.add_argument('--sample_level', type=int, default=3, help='number of sub series')
    parser.add_argument('--group', type=int, default=4, help='number of group')
    parser.add_argument('--shuffle_channel', type=int, default=4, help='number of shuffled channel')

    parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=20, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='MSE', help='loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

    # de-stationary projector params
    parser.add_argument('--p_hidden_dims', type=int, nargs='+', default=[128, 128], help='hidden layer dimensions of projector (List)')
    parser.add_argument('--p_hidden_layers', type=int, default=2, help='number of hidden layers in projector')

    # mra
    parser.add_argument('--out_mra', type=bool, default=True, help='out mra result')


    args = parser.parse_args()
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    if args.use_gpu and args.use_multi_gpu:
        args.dvices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    print('Args in experiment:')
    print(args)
    

    train_data, train_loader = data_provider(args, 'train')
    val_data, val_loader = data_provider(args, 'val')


    model = TimeSeriesLightningModule(args)
    

    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath='checkpoints/',
        filename=f'{args.model}-{args.data}-{{epoch:02d}}-{{train_loss:.2f}}',
        save_top_k=3,
        mode='min'
    )
    
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        patience=args.patience,
        mode='min'
    )
    

    logger = TensorBoardLogger('lightning_logs', name=f'{args.model}_{args.data}')
    

    trainer = pl.Trainer(
        max_epochs=100,
        callbacks=[checkpoint_callback, early_stop_callback],
        logger=logger
    )
    

    trainer.fit(
        model,
        train_dataloader=train_loader
    )
    

if __name__ == '__main__':
    main()
