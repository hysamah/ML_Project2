from course_helpers import *
from helpers import *
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import torch
from torch.nn import Sequential, Module, LSTM, ReLU, Linear, Sigmoid, Dropout, Embedding
import torchvision
# PyTorch TensorBoard support
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import torch.nn.functional as F
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import csv
def create_csv_submission(ids, y_pred, name):
    """
    Creates an output file in .csv format for submission to Kaggle or AIcrowd
    Arguments: ids (event ids associated with each prediction)
               y_pred (predicted class labels)
               name (string name of .csv output file to be created)
    """
    with open(name, "w") as csvfile:
        fieldnames = ["Id", "Prediction"]
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({"Id": int(r1)+1, "Prediction": int(r2)})


def train_one_epoch(epoch_index, tb_writer):
    running_loss = 0.
    last_loss = 0.
    running_accuracy = 0.0

    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting

    with tqdm(training_loader, unit="batch") as tepoch:
        for i, data in enumerate(tepoch):
            tepoch.set_description(f"Epoch {epoch_index}")


    #for i, data in enumerate(tqdm(training_loader), unit= "batches", ):
            # Every data instance is an input + label pair
            inputs, labels = data

            # Zero your gradients for every batch!
            optimizer.zero_grad()


            # Make predictions for this batch
            outputs = model(inputs)
            accuracy  = accuracy_score(labels.cpu(), [1 if x > 0.5 else 0 for x in outputs.cpu()])
            # Compute the loss and its gradients
            loss = loss_fn(outputs, labels)
            loss.backward()

            # Adjust learning weights
            optimizer.step()
            # Gather data and report
            running_loss += loss.item()
            running_accuracy += accuracy

            tepoch.set_postfix(loss=loss.item(), accuracy = accuracy)
            if i % batch_size == batch_size-1:
                last_loss = running_loss / batch_size # loss per batch
                #print('  epoch {} batch {} loss: {}'.format(epoch_index, (i + 1)/batch_size, last_loss))
                tb_x = epoch_index * len(training_loader) + i + 1
                #print(tb_x)
                tb_writer.add_scalar('Loss/train', last_loss, global_step = tb_x)
                running_loss = 0.

    return last_loss, running_accuracy/i


class make_torch_dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels=[]):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        inp = self.encodings[idx]
        label = self.labels[idx]
        return inp, label

    def __len__(self):
        return len(self.encodings)

class make_torch_testset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        inp = self.encodings[idx]
        return inp

    def __len__(self):
        return len(self.encodings)

class lstm_classifier(Module):
    def __init__(self, num_embeddings, embed_dim, glove_embd, max_len) -> None:
        super().__init__()
        self.emd =  Embedding(num_embeddings= num_embeddings, embedding_dim= embed_dim, _weight=glove_embd)
        self.lstm1 = LSTM(input_size = embed_dim, hidden_size = 256, bidirectional = True, batch_first  = True)
        self.lstm2 = LSTM(input_size = 512, hidden_size = 150, bidirectional = True,  batch_first  = True)
        self.fc1 = Linear(300,128)
        self.drp = Dropout(0.1)
        self.fc2 = Linear(128, 64)
        self.fc3 = Linear(64,1)
        self.reg = Sigmoid()


    def forward(self, x):
        x = self.emd(x)
        x = x.to(torch.float32)
        x, states = self.lstm1(x)
        x, states = self.lstm2(x)
        x = F.relu(self.fc1(x[:,0]))
        x = self.drp(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.reg(x)
        x = x.reshape(x.shape[0])

        return x

##data preprocessing 
DATA_PATH = 'twitter-datasets'
Dataset = read(DATA_PATH)


embd = 'embeddings'
embedding = np.load(embd+'.npy')
glove_embd = embedding


test_set = pd.read_csv('test_set_token.csv')
train_set = pd.read_csv('train_set_token.csv', )
num_embeddings  = glove_embd.shape[0]
embed_dim = glove_embd.shape[1]
max_len = train_set.shape[1]-1
batch_size = 16
n_batches = train_set.shape[0]/batch_size

X_train = train_set.loc[:, train_set.columns!='sentiment']
Y_train = train_set['sentiment']
device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda:1')
glove_embd = torch.tensor(glove_embd)

data_split = (X_train.shape[0]*0.2)%batch_size
data_split  = X_train.shape[0]*0.2 - data_split
data_split=data_split/2
remd = -data_split + X_train.shape[0] - data_split
addedv = remd%data_split

X_val =  torch.tensor(pd.concat([X_train[:int(data_split)], X_train[-int(data_split):]],  axis=0, ignore_index=True).reset_index(drop=True).to_numpy()).to(device)
Y_val =  torch.tensor(pd.concat([Y_train[:int(data_split)], Y_train[-int(data_split):]],  axis=0, ignore_index=True).reset_index(drop=True).to_numpy(dtype='float32')).to(device)
X_train = torch.tensor(X_train[int(data_split+addedv):-int(data_split)].reset_index(drop=True).to_numpy()).to(device)
Y_train =  torch.tensor(Y_train[int(data_split+addedv):-int(data_split)].reset_index(drop=True).to_numpy(dtype='float32')).to(device)
X_test =  torch.tensor(test_set.to_numpy()).to(device)




train_dataset = make_torch_dataset(X_train, Y_train)
val_dataset = make_torch_dataset(X_val, Y_val)
test_dataset = make_torch_testset(X_test)

# Create data loaders for our datasets; shuffle for training, not for validation
training_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
validation_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

# Report split sizes
print('Training set has {} instances'.format(len(train_dataset)))
print('Validation set has {} instances'.format(len(val_dataset)))
print('Test set has {} instances'.format(len(test_dataset)))



# initializing model and losses
model = lstm_classifier(num_embeddings, embed_dim, glove_embd, max_len).to(device)
loss_fn = torch.nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# starting main training loop
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
writer = SummaryWriter('runs/fashion_trainer_{}'.format(timestamp))

EPOCHS = 5

best_vloss = 1_000_000.



for epoch in range(EPOCHS):
    print('EPOCH {}:'.format(epoch + 1))

    # Make sure gradient tracking is on, and do a pass over the data
    model.train(True)
    avg_loss, acc = train_one_epoch(epoch, writer)

    # # We don't need gradients on to do reporting
    model.train(False)

    running_vloss = 0.0
    running_vacc = 0.0
    for i, vdata in enumerate(validation_loader):
        vinputs, vlabels = vdata
        voutputs = model(vinputs).detach()
        vacc  = accuracy_score(vlabels.cpu(), [1 if x > 0.5 else 0 for x in voutputs.cpu()])
        running_vacc += vacc
        vloss = loss_fn(voutputs, vlabels)
        running_vloss += vloss

    avg_vloss = running_vloss / (i + 1)
    avg_vacc = running_vacc / (i + 1)
    print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))

    # Log the running loss averaged per batch
    # for both training and validation
    writer.add_scalars('Training vs. Validation/Loss',
                    { 'Training' : avg_loss, 'Validation' : avg_vloss },
                    epoch + 1)
    writer.add_scalars('Training vs. Validation/Accuracy',
                    { 'Training' : acc, 'Validation' : avg_vacc },
                    epoch + 1)
    #writer.flush()

    # Track best performance, and save the model's state
    if avg_vloss < best_vloss:
        best_vloss = avg_vloss
        model_path = 'model_{}_{}'.format(timestamp, epoch)
        torch.save(model.state_dict(), model_path)




model.train(False)
preds_test = []
for i, vdata in enumerate(test_loader):
    vinputs = vdata
    voutputs = model(vinputs).detach()
    preds_test.extend(voutputs.cpu())
preds = np.ones_like(preds_test)
preds[np.where(np.array(preds_test)<0.5)] = -1
preds = pd.DataFrame(preds)

create_csv_submission(preds.index.values.tolist(), np.asarray(preds), 'model_{}_.csv'.format(timestamp))