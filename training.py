import argparse
import time
import pandas as pd

from models.gcn import GCNNet
from utils import *
from models.cvae_models import VAE
import torch.nn as nn
import torch.nn.functional as F

# training function at each epoch
def train(model, device, train_loader, optimizer, epoch, vae, indice, BATCH_SIZE, node, edge, feature_t):
    print('Training on {} samples...'.format(len(train_loader.dataset)))
    model.train()
    for batch_idx, data in enumerate(train_loader):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = start_idx + BATCH_SIZE
        idx = indice[start_idx:end_idx]
        batch_ids = [train_id[i] for i in idx]
        for j, key in enumerate(batch_ids):
            node[j] = representations[key].squeeze()
            edge[j] = torch.tensor(concat[key], dtype=torch.float32).squeeze()
            target = torch.from_numpy(feature[key])
            if feature[key].shape[1] < 1024:
                padding = (0, 0, 0, 1024 - feature[key].shape[1])
                target = F.pad(target, pad=padding, mode='constant', value=0)
            feature_t[j] = target.squeeze()
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data, vae, node.clone(), edge.clone(), feature_t.clone())
        loss = loss_fn(output, data.y.view(-1, 1).float().to(device))
        loss.backward()
        optimizer.step()
        if batch_idx % LOG_INTERVAL == 0:
            print('Train epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch,
                                                                           batch_idx * len(data.x),
                                                                           len(train_loader.dataset),
                                                                           100. * batch_idx / len(train_loader),
                                                                           loss.item()))


def predicting(model, device, loader, vae, indice, BATCH_SIZE, node, edge,feature_t):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    print('Make prediction for {} samples...'.format(len(loader.dataset)))
    with torch.no_grad():
        for batch_idx, data in enumerate(loader):
            start_idx = batch_idx * BATCH_SIZE
            end_idx = start_idx + BATCH_SIZE
            idx = indice[start_idx:end_idx]
            batch_ids = [test_id[i] for i in idx]
            for j, key in enumerate(batch_ids):
                node[j] = representations[key].squeeze()
                edge[j] = torch.tensor(concat[key], dtype=torch.float32).squeeze()
                target = torch.from_numpy(feature[key])
                if feature[key].shape[1] < 1024:
                    padding = (0, 0, 0, 1024 - feature[key].shape[1])
                    target = F.pad(target, pad=padding, mode='constant', value=0)
                feature_t[j] = target.squeeze()
            data = data.to(device)
            output = model(data, vae, node.clone(), edge.clone(), feature_t.clone())
            total_preds = torch.cat((total_preds, output.cpu()), 0)
            total_labels = torch.cat((total_labels, data.y.view(-1, 1).cpu()), 0)
    return total_labels.numpy().flatten(), total_preds.numpy().flatten()


# loss_fn = nn.MSELoss(reduce=False)  # reduce = False，返回向量形式的 loss　
# loss_fn1 = nn.MSELoss()
loss_fn = nn.MSELoss()
LOG_INTERVAL = 20


def main(args):
    dataset = args.dataset
    modeling = [GCNNet]
    # modeling = [GraphConvNet]
    model_st = modeling[0].__name__

    cuda_name = "cuda:0"
    print('cuda_name:', cuda_name)

    TRAIN_BATCH_SIZE = args.batch_size
    TEST_BATCH_SIZE = args.batch_size
    LR = args.lr

    NUM_EPOCHS = args.epoch

    print('Learning rate: ', LR)
    print('Epochs: ', NUM_EPOCHS)

    # Main program: iterate over different datasets
    print('\nrunning on ', model_st + '_' + dataset)
    processed_data_file_train = 'data/processed/' + dataset + '_train.pt'
    processed_data_file_test = 'data/processed/' + dataset + '_test.pt'
    if ((not os.path.isfile(processed_data_file_train)) or (not os.path.isfile(processed_data_file_test))):
        print('please run create_data.py to prepare data in pytorch format!')
    else:
        train_data = TestbedDataset(root='data', dataset=dataset + '_train')
        test_data = TestbedDataset(root='data', dataset=dataset + '_test')
        # make data PyTorch mini-batch processing ready
        train_loader = DataLoader(train_data, batch_size=TRAIN_BATCH_SIZE, shuffle=False, drop_last=True)
        test_loader = DataLoader(test_data, batch_size=TEST_BATCH_SIZE, shuffle=False, drop_last=True)
        # training the model
        device = torch.device(cuda_name if torch.cuda.is_available() else "cpu")
        # model = modeling[0](k1=1,k2=2,k3=3,embed_dim=128, num_layer=1, device=device).to(device)
        node = torch.zeros(TRAIN_BATCH_SIZE, 1024, 1280, dtype=torch.float32).cuda()
        edge = torch.zeros(TRAIN_BATCH_SIZE, 1024, 1024, dtype=torch.float32).cuda()
        feature_t = torch.zeros(TRAIN_BATCH_SIZE, 1024, 1280, dtype=torch.float32).cuda()
        vae = VAE(
            encoder_layer_sizes=args.encoder_layer_sizes,
            latent_size=args.latent_size,
            decoder_layer_sizes=args.decoder_layer_sizes).to(device)
        model = modeling[0](k1=1, k2=2, k3=3, embed_dim=128, num_layer=1, device=device).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        best_mse = 1000
        best_ci = 0
        best_epoch = -1
        # model_file_name = 'model' + model_st + '_' + dataset +  '.model'
        result_file_name = 'result' + model_st + '_' + dataset + '.csv'

        for epoch in range(NUM_EPOCHS):
            start_time = time.time()
            indices_train = list(iter(train_loader.sampler))
            indices_test = list(iter(test_loader.sampler))
            train(model, device, train_loader, optimizer, epoch + 1, vae, indices_train, TRAIN_BATCH_SIZE, node, edge, feature_t)
            G, P = predicting(model, device, test_loader, vae, indices_test, TEST_BATCH_SIZE, node, edge, feature_t)
            ret = [rmse(G, P), mse(G, P), pearson(G,P),spearman(G,P), ci(G, P), get_rm2(G.reshape(G.shape[0], -1), P.reshape(P.shape[0], -1))]
            if ret[1] < best_mse:
                if args.save_file:
                    model_file_name = args.save_file + '.model'
                    torch.save(model.state_dict(), model_file_name)

                with open(result_file_name, 'w') as f:
                    f.write('rmse,mse,pearson,spearman,ci,rm2\n')
                    f.write(','.join(map(str, ret)))
                best_epoch = epoch + 1
                best_mse = ret[1]
                best_ci = ret[-2]
                print('rmse improved at epoch ', best_epoch, '; best_mse,best_ci:', best_mse, best_ci, model_st,
                      dataset)
            else:
                print(ret[1], 'No improvement since epoch ', best_epoch, '; best_mse,best_ci:', best_mse, best_ci,
                      model_st, dataset)
            end_time = time.time()  # End time for the epoch
            epoch_time = end_time - start_time
            print('Time taken for epoch {}: {:.2f} seconds'.format(epoch + 1, epoch_time))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run DeepGLSTM")

    parser.add_argument("--dataset", type=str, default='DTC', help="Dataset Name (davis,kiba,DTC,Metz,ToxCast,Stitch)")

    parser.add_argument("--epoch",
                        type=int,
                        default=1000,
                        help="Number of training epochs. Default is 1000."
                        )

    parser.add_argument("--lr",
                        type=float,
                        default=0.0005,
                        help="learning rate",
                        )

    parser.add_argument("--batch_size", type=int,
                        default=128,
                        help="Number of drug-target per batch. Default is 128 for davis.")  # batch 128 for Davis

    parser.add_argument("--save_file", type=str,
                        default='./pretrained_model/DTC',
                        help="Where to save the trained model. For example davis.model")
    parser.add_argument("--concat", type=int, default=1)
    parser.add_argument("--encoder_layer_sizes", type=list, default=[128, 256])
    parser.add_argument("--decoder_layer_sizes", type=list, default=[256, 78])
    parser.add_argument("--latent_size", type=int, default=10)
    args = parser.parse_args()
    representations = np.load('data/node/' + args.dataset + '.npz', allow_pickle=True)
    representations = representations['dict'][()]
    concat = np.load('data/edge/' + args.dataset + '.npz', allow_pickle=True)
    concat = concat['dict'][()]
    feature = np.load('data/' + args.dataset + '.npz', allow_pickle=True)
    feature = feature['dict'][()]
    df1 = pd.read_csv('data/' + args.dataset + '_train.csv')
    train_id = list(df1['protein_id'])
    df2 = pd.read_csv('data/' + args.dataset + '_test.csv')
    test_id = list(df2['protein_id'])
    print(args)
    main(args)
