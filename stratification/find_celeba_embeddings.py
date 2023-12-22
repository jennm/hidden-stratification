import argparse
import json
import torch
import pandas as pd


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


tail_cache = dict()
def hook_fn(module, input, output):
    # print(f"module type: {type(module)}\tinput type: {type(input)}\toutput type: {type(output)}")
    # device = module.get_device()
    device = output.get_device()
    if device in tail_cache:
        tail_cache[device].append(input[0].clone().detach())
    else:
        tail_cache[device] = [input[0].clone().detach()]
    print(len(tail_cache[device]), module)


def get_hooks(model):
    hooks = []
    num_layers = sum(1 for _ in model.modules())
    print('model num layers', num_layers)
    for i, module in enumerate(model.modules()):
        if i >= num_layers - 5:
            print(f"{i}: {num_layers - i}")
            print(module)
            hook = module.register_forward_hook(hook_fn)
            hooks.append(hook)

    print('hooks_length: ', len(hooks))


def write_ds(model, dataloader, data_info_path):
    global tail_cache
    
    criterion = torch.nn.CrossEntropyLoss(reduction='none') 
    data = [[] for _ in range(5)]
    outputs = {
        '1-to-last': [],
        '2-to-last': [],
        '3-to-last': [],
        '4-to-last': [],
        '5-to-last': [],
        '6-to-last': [],
        '7-to-last': [],
        'protected-class': [],
        'loss': [],
        'actual_label': [],
        'predicted_label': [],
    }

    data_info_df = pd.read_csv(data_info_path)

    loc = 0
    cols = list(data_info_df.columns)[0].split(' ')

    blond_hair_ind = 0
    male_ind = 0
    i = 0
    while blond_hair_ind * male_ind == 0:
        if cols[i] == 'Blond_Hair':
            blond_hair_ind = i
        elif cols[i] == 'Male':
            male_ind = i
        i += 1

    for example in dataloader:
        inputs = example[0] 
        batch_size = len(inputs)
        y_true = list()
        for i in range(batch_size):
            col_list = list(data_info_df.iloc[loc + i])[0].split(' ')
            while '' in col_list:
                col_list.remove('')
            y = int(col_list[blond_hair_ind])
            y = max(0, y)
            y_true.append(y)
            male = int(col_list[male_ind])
            if y == 1:
                if male:
                    # blond male
                    outputs['protected-class'].append(0)
                else:
                    # blond female
                    outputs['protected-class'].append(1)
            else:
                if male:
                    # brunette male
                    outputs['protected-class'].append(2)
                else:
                    # brunette female
                    outputs['protected-class'].append(3)
            outputs['actual_label'].append(y)


        print('before through model', len(tail_cache))
        output = model(inputs)
        print('after through model', len(tail_cache))
        pred = torch.argmax(output, 1)
        pred = pred.to(torch.float)
        y_true_tensor = torch.tensor(y_true, dtype=torch.float)
        loss = criterion(pred, y_true_tensor)
        for i in range(batch_size):
            outputs['predicted_label'].append(int(pred[i].item()))
            outputs['loss'].append(loss.item())

        tail_cache_keys = list(tail_cache.keys())
        for key in tail_cache_keys:
            for idx, tensor in enumerate(tail_cache[key]):
                batch = list(torch.split(tensor, 1, dim=0))
                outputs[f'{idx + 1}-to-last'].extend(batch)
            
        loc += batch_size
        break
        # if loc > batch_size * 5:
        #     break
        tail_cache = dict()

    # print(outputs)
    return
    # torch.save(outputs, f'celeba_embeddings.pt')


def main():
    parser = argparse.ArgumentParser(description='Caches model run tensors.')
    parser.add_argument('--config_fp', type=str, help='relative file path to config json.')
    parser.add_argument('--pretrained_fp', type=str, help='relative file path to model checkpoint.')
    parser.add_argument('--data_info', type=str, help='relative file path to the data info')
    args = parser.parse_args()

    harness, dataloaders, num_classes, model = setup(args.config_fp)
    
    state_dict = torch.load(args.pretrained_fp, map_location=torch.device('cpu'))
    # print(sate_dict['state_dict'].keys())
    # return
    model.load_state_dict(state_dict['state_dict'], strict=False)
    model.eval()
    print('before get_hooks', len(tail_cache))
    get_hooks(model)
    print('after get_hooks', len(tail_cache))
    # print('before')
    write_ds(model, dataloaders['train'], args.data_info)
    # print('after')

if __name__ == '__main__':
    main()