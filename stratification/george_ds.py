import argparse
import json
import torch


from stratification.utils.parse_args import get_config
from stratification.harness import GEORGEHarness

def setup(config_fp):
    with open(config_fp, 'r') as f:
        config = json.dumps(json.load(f))
    config = get_config([config])

    harness = GEORGEHarness(config, use_cuda=torch.cuda.is_available(), log_format='simple')
    dataloaders = harness.get_dataloaders(config, mode='erm')
    num_classes = dataloaders['train'].dataset.get_num_classes('superclass')
    model = harness.get_nn_model(config, num_classes=num_classes, mode='erm')

    return harness, dataloaders, num_classes, model


tail_cache = []
def hook_fn(module, input, output):
    tail_cache.append(input[0].clone().detach())


def get_hooks(model):
    hooks = []
    num_layers = sum(1 for _ in model.modules())
    print('model num layers', num_layers)
    for i, module in enumerate(model.modules()):
        if i >= num_layers - 5:
            hook = module.register_forward_hook(hook_fn)
            hooks.append(hook)


def write_ds(model, dataloader):
    global tail_cache
    
    data = [[] for _ in range(5)]

    for example in dataloader:
        input = example[0] 
        model(input)

        for idx, tensor in enumerate(tail_cache):
            batch = list(torch.split(tensor, 1, dim=0))
            data[idx].extend(batch)
        
        tail_cache = []

    for i, layer in enumerate(data):
        outputs = torch.stack(layer)
        torch.save(outputs, f'{i}_layer_outputs.pt')


def main():
    parser = argparse.ArgumentParser(description='Caches model run tensors.')
    parser.add_argument('--config_fp', type=str, help='relative file path to config json.')
    parser.add_argument('--pretrained_fp', type=str, help='relative file path to model checkpoint.')
    args = parser.parse_args()

    harness, dataloaders, num_classes, model = setup(args.config_fp)
    
    state_dict = torch.load(args.pretrained_fp)#, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict['state_dict'], strict=False)
    model.eval()
    get_hooks(model)
    write_ds(model, dataloaders['test'])

if __name__ == '__main__':
    main()