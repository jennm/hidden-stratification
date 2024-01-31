class GroupDRO:
    '''
    what does classify.train do
    '''

    def __init__(self):
        pass
    

    def run_epoch(model, optimizer, criterion, dataloader, train: bool, classifiers):
        if train: model.train()
        else: model.eval()
        with torch.set_grad_enabled(train):
            for batch_idx, (inputs, targets) in enumerate(dataloader):
                N = len(inputs)
                superclass_targets = None  # idk how we will get these
                
                for classifier in classifiers:
                    predicted_groups = torch.argmax(classifier(inputs), 1)

                logits = model(inputs)
                co = criterion(logits, ground_truth, group_ids)
                # loss, per sample loss, corrects, _
                loss, (losses, corrects), _  = co
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        pass


    def train(self, epochs):
        criterion = None
        optimizer = None

        for epoch in range(epochs):
            # run training epoch
            # run validation epoch

            pass

        # save model