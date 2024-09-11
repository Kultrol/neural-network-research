import torch


class Neural_Network(torch.nn.Module):

    # Constructor
    def __init__(self, b_size=100, l_rate=1, percentage=0, directory='file.csv'):
        super().__init__()

        self.file = open(directory, "w")
        INPUT_SIZE = 28 * 28
        OUTPUT_SIZE = 10
        self.linlayer = torch.nn.Linear(INPUT_SIZE, OUTPUT_SIZE, bias=False)
        self.model = torch.nn.Sequential(self.linlayer)

        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adadelta(self.model.parameters(), lr=l_rate)

        self.b_size = b_size
        self.percentage = percentage

        self.flatten = torch.nn.Flatten()
        self.clear()

        self.ce_asy_list = [0]
        self.a_list = [0]
        self.b_list = [0]
        self.stop_counter = 0
        self.ces = []
        self.stop_training = False

    # Manually initialize weight matrices to ones
    def clear(self) -> None:
        with torch.no_grad():
            for _, param in self.model.named_parameters():
                if param.requires_grad:
                    param.copy_(torch.ones_like(param))

    def forward(self, x):
        return (self.model(self.flatten(x)))