import torch
import matplotlib.pyplot as plt
from pl_run import TimeSeriesLightningModule, get_fixed_subset
from data_provider.data_factory import data_provider
from torch.utils.data import DataLoader
import argparse
import numpy as np
import seaborn as sns


plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 15



def visualize_predictions(batch_x, batch_y, outputs):

    plt.figure(figsize=(12, 6))
    

    full_true = torch.cat((batch_x[0, :, -1], batch_y[0, :, -1]), dim=0).cpu().numpy()

    plt.plot(range(len(full_true)), full_true, color='black', label='True Data')

    plt.plot(range(batch_x.shape[1], batch_x.shape[1] + batch_y.shape[1]), 
             outputs[0, :, 0].detach().cpu().numpy(), color='orange', label='Prediction')

    input_length = batch_x.shape[1]
    arrow_y = plt.ylim()[0] - (plt.ylim()[1] - plt.ylim()[0]) * 0.1
    gap_size = input_length * 0.3 
    left_arrow_end = input_length/2 - gap_size/2
    right_arrow_start = input_length/2 + gap_size/2

    plt.annotate('', xy=(0, arrow_y), xytext=(left_arrow_end, arrow_y),
                arrowprops=dict(arrowstyle='->', color='red', lw=2))
    plt.annotate('', xy=(input_length, arrow_y), xytext=(right_arrow_start, arrow_y),
                arrowprops=dict(arrowstyle='->', color='red', lw=2))
    plt.vlines(0, arrow_y, full_true[0], linestyles='dashed', colors='black')
    plt.vlines(input_length, arrow_y, full_true[input_length-1], linestyles='dashed', colors='black')
    

    plt.text(input_length/2, arrow_y, 'model input', 
             horizontalalignment='center',
             verticalalignment='center',
             color='black',
             fontsize=20,
             fontfamily='Times New Roman')
    
    plt.legend()
    plt.title('Model Prediction(A Case Study)')
    plt.xlabel('Time Steps')
    plt.ylabel('Water Level Value(m)')
    plt.grid(True)
  
    current_ylim = plt.ylim()
    plt.ylim(arrow_y - (plt.ylim()[1] - plt.ylim()[0]) * 0.05, current_ylim[1])
    
    plt.tight_layout()
    plt.show()



def visualize_projection_output(dec_out, station_names=None):
    # dec_out shape: [B, T, D]
    relation_matrix = torch.sigmoid(dec_out[0]).cpu().detach().numpy()  # [T, D]
    
    if station_names is None:
        station_names = [f'Station {i+1}' for i in range(relation_matrix.shape[1])]
    
    plt.figure(figsize=(12, 8))
    
    T = relation_matrix.shape[0]
    selected_timesteps = [0, T//2, T-1]  
    y_labels = [f'{i}' for i in range(T)]
    shown_labels = ['' if i not in selected_timesteps else y_labels[i] for i in range(T)]
    heatmap = sns.heatmap(relation_matrix, 
                         cmap='RdYlBu_r', 
                         xticklabels=station_names,
                         yticklabels=shown_labels, 
                         center=0.5,  
                         vmin=0, vmax=1, 
                         cbar_kws={'label': 'Relationship Strength'})
    
    plt.title('Temporal-Spatial Relationship Matrix')
    plt.xlabel('The Acronym of the Station Name')
    plt.ylabel('Time Steps')
    plt.tight_layout()
    plt.show()




def load_model_and_get_mra(args, checkpoint_path, data_loader):

    model = TimeSeriesLightningModule.load_from_checkpoint(checkpoint_path, args=args)
    model.eval()


    batch = next(iter(data_loader))
    batch_x, batch_y, batch_x_mark, batch_y_mark = batch
    batch_x = batch_x.float()
    batch_x_mark = batch_x_mark.float()


    dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float()
    dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float()

    with torch.no_grad():
        enc_out = model.model.enc_embedding(batch_x, batch_x_mark)
        enc_out = model.model.predict_linear(enc_out.permute(0, 2, 1)).permute(0, 2, 1)

        mra, conv_out = model.model.model[0](enc_out, return_mra=True, return_conv=True)
    
        temp_enc_out = enc_out.clone()
        for i in range(model.model.layer):
            temp_enc_out = model.model.layer_norm(model.model.model[i](temp_enc_out))
            
        projection_out = model.model.projection(temp_enc_out)
        print("Projection output shape:", projection_out.shape)

        model_output = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        f_dim = -1 if args.features == 'MS' else 0
        model_output = model_output[:, -args.pred_len:, f_dim:]
        batch_y = batch_y[:, -args.pred_len:, f_dim:]
        
        print("Model prediction output shape:", model_output.shape)

    return mra, conv_out, batch_x, projection_out,batch_y, model_output


def visualize_conv_output(conv_out,selected_channels):
    total_channels = conv_out.shape[1]
    for idx, channel_idx in enumerate(selected_channels):
        plt.figure(figsize=(12, 8))
        feature_map = conv_out[0, channel_idx, :, :].cpu().numpy()
        plt.imshow(new_feature_map, aspect='auto', cmap='plasma_r')
        plt.colorbar(label='Feature Value')
        plt.title(f'2D-temporal variation representation of channel {idx + 1}')
        plt.xlabel('Adaptive MRA Level')
        plt.ylabel('Time Step')
        plt.xticks(range(6), ['1', '2', '3', '4', '5', '6'])
        plt.close() 


def visualize_mra_with_variance(mra, input_data):   
    num_variables = input_data.shape[-1]
    total_timesteps = mra.shape[1]  

    for var_idx in range(num_variables):
        plt.figure(figsize=(15, 12))
        levels = mra.shape[-1] 

        plt.subplot(levels, 1, 1)
        original_signal = input_data[0, :, var_idx].cpu().numpy()
        plt.plot(original_signal)
        plt.title(f'Original Time Series of {station_names[var_idx]}')
        plt.grid(True)
        subplot_idx = 2  
        level_num = 1  
        for i in range(levels):  
            mra_level = mra[0, mra_timestep_idx, :, i].cpu().numpy()
            
            plt.subplot(levels, 1, subplot_idx)
            plt.plot(mra_level)
            plt.title(f'Adaptive MRA Level {level_num}')
            plt.grid(True)
            
            subplot_idx += 1
            level_num += 1
        plt.tight_layout()
        plt.close() 



def main():
  parser = argparse.ArgumentParser(description='han chuan water level prediction')

  # basic config
  parser.add_argument('--task_name', type=str, default='long_term_forecast',
                      help='task name')
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


  checkpoint_path = 'checkpoints/DLGNet-hj-epoch=51-val_loss=0.00001.ckpt'

  test_data, _ = data_provider(args, 'test')
  fixed_test_data = get_fixed_subset(test_data, num_samples=1)
  test_loader = DataLoader(fixed_test_data, batch_size=1, shuffle=False)


  mra, conv_out, input_data, projection_out, batch_y, model_predictions = load_model_and_get_mra(args, checkpoint_path, test_loader)
  
  print("Input data shape:", input_data.shape)
  print("MRA shape:", mra.shape)
  print("Conv output shape:", conv_out.shape)
  visualize_mra_with_variance(mra, input_data)
  visualize_conv_output(conv_out)
  station_names = ['HJG', 'XY', 'YC', 'HZ', 
                  'SY', 'YK', 'XT', 'HC']
  visualize_projection_output(projection_out, station_names)
  visualize_predictions(input_data, batch_y, model_predictions)



if __name__ == '__main__':
    main()
