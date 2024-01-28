from torch.nn import Module, LSTM, Linear
class Net(Module):
    def __init__(self,config):#创建一个类
        super(Net, self).__init__()#首先找到Net的父类（比如是类NNet)，然后把类Net的对象self转换为类NNet的对象，然后"被转换"的类
        self.lstm=LSTM(input_size=config.input_size,
                       hidden_size=config.hidden_size,
                       num_layers=config.lstm_layers,
                       batch_first=True,
                       dropout=config.dropout_rate,
                       bidirectional=config.bidirectional)

        self.linear=Linear(in_features=config.hidden_size,out_features=config.output_size)
    def forward(self,x,hidden=None):
        lstm_out,hidden=self.lstm(x,hidden)
        linear_out=self.linear(lstm_out)
        return linear_out,hidden