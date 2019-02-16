from cnn import *
from solver import *
import pickle
import matplotlib.pyplot as plt


with open('mnist.pkl','rb') as f:
    raw_data = pickle.load(f,encoding='latin1')
    
train_set, val_set, test_set = raw_data

X_train, y_train = train_set
X_val, y_val     = val_set
X_test, y_test   = test_set

idx_train = np.random.randint(50000,size = 5000) 
idx_val = np.random.randint(10000,size = 1000) 
idx_test = np.random.randint(10000,size = 1000) 

X_train = X_train[idx_train]
y_train = y_train[idx_train]

X_val = X_val[idx_val]
y_val = y_val[idx_val]

X_test = X_test[idx_val]
y_test = y_test[idx_val]

N_train = X_train.shape[0]
N_val   = X_val.shape[0]
N_test  = X_test.shape[0]

X_train = X_train.reshape((N_train, 1, 28, 28))
X_val = X_val.reshape((N_val, 1, 28, 28))
X_test = X_test.reshape((N_test, 1, 28, 28))

data = {
        'X_train': X_train,
        'y_train': y_train,
        'X_val': X_val,
        'y_val': y_val,
        }

model = ConvNet(num_filters=32, filter_size=7, hidden_dim=100, weight_scale=1e-3, reg=0.0)


solver = Solver(model, data,
              #update_rule='rmsprop',
              update_rule='sgd_momentum',
              optim_config={
                'learning_rate': 1e-2,
              },
              lr_decay=0.99995,
              num_epochs=10, batch_size=100,
              print_every=100)

solver.train()

test_acc = solver.check_accuracy(X_test, y_test)
print(test_acc)

plt.plot(solver.val_acc_history, label="val_acc")
plt.plot(solver.train_acc_history, label="train_acc")
plt.legend(loc="lower right")
