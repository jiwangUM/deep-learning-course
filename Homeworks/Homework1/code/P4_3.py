from softmax import *
from solver import *
import pickle
import matplotlib.pyplot as plt


with open('mnist.pkl','rb') as f:
    raw_data = pickle.load(f,encoding='latin1')
    
train_set, val_set, test_set = raw_data

X_train, y_train = train_set
X_val, y_val     = val_set
X_test, y_test   = test_set

N_train = X_train.shape[0]
N_val   = X_val.shape[0]
N_test  = X_test.shape[0]

data = {
        'X_train': X_train,
        'y_train': y_train,
        'X_val': X_val,
        'y_val': y_val,
        }

model = SoftmaxClassifier(input_dim=X_train.shape[1], hidden_dim=None, weight_scale=1e-2, reg=0.0)


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

#(Epoch 10 / 10) train acc: 0.925000; val_acc: 0.920900
#0.9179
#
#(Epoch 20 / 10) train acc: 0.931000; val_acc: 0.925200
#0.9213
#
#(Epoch 30 / 10) train acc: 0.949000; val_acc: 0.926000
#0.922
#
#(Epoch 40 / 10) train acc: 0.930000; val_acc: 0.927900
#0.9239
#
#(Epoch 50 / 10) train acc: 0.933000; val_acc: 0.927500
#0.9233
#
#(Epoch 60 / 10) train acc: 0.924000; val_acc: 0.928300
#0.9235
#
#(Epoch 70 / 10) train acc: 0.926000; val_acc: 0.929600
#0.9242
#
#(Epoch 80 / 10) train acc: 0.937000; val_acc: 0.929700
#0.9244
#
#(Epoch 90 / 10) train acc: 0.936000; val_acc: 0.928700
#0.9246
#
#(Epoch 100 / 10) train acc: 0.927000; val_acc: 0.929500
#0.9247
#
#(Epoch 110 / 10) train acc: 0.922000; val_acc: 0.929300
#0.9254
#
#(Epoch 120 / 10) train acc: 0.941000; val_acc: 0.929600
#0.9251
#
#(Epoch 130 / 10) train acc: 0.929000; val_acc: 0.928900
#0.9268
#
#(Epoch 140 / 10) train acc: 0.925000; val_acc: 0.930800
#0.9249
#(Iteration 4901 / 5000) loss: 0.284574
#(Epoch 150 / 10) train acc: 0.940000; val_acc: 0.930000
#0.9272