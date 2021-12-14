import torch.nn as nn

from models.yolo import Model

class DynamicNN(nn.Module):
    def __init__(self, parallel_pipe, cfg, device):
        super().__init__()
        self.model_list = []
        for i in range(parallel_pipe):
            model = Model(cfg).to(device)
            self.model_list.append(model)

    def forward(self, x, action=None):
        output = []
        if action is None:
            action = []
        for idx, model in enumerate(self.model_list):
            if idx in action:
                output.append(model(x))
            else:
                output.append(None)

        return output

if __name__ == '__main__':
    model = DynamicNN(parallel_pipe=5, cfg='yolov5n_feature.yaml', device='cpu')
