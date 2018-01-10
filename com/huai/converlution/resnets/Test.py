from com.huai.converlution.resnets.resnets_utils import load_dataset
from com.huai.converlution.resnets.resnets_utils import process_orig_datasets

path = 'D:/LiangYiHuai/deepleanring/resnets/datasets'
orig_data = load_dataset(path)

X_train, Y_train, X_test, Y_test = process_orig_datasets(orig_data)

print ("number of training examples = " + str(X_train.shape[0]))
print ("number of test examples = " + str(X_test.shape[0]))
print ("X_train shape: " + str(X_train.shape))
print ("Y_train shape: " + str(Y_train.shape))
print ("X_test shape: " + str(X_test.shape))
print ("Y_test shape: " + str(Y_test.shape))

