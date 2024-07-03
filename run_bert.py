from Bert import *

if __name__ == '__main__':

    MAX_LEN = 256
    BATCH_SIZE = 16
    NUM_EPOCHS = 5

    trainData = load_dataset('data/train.txt')
    valData = load_dataset('data/dev.txt')

    trainDataset = myDataset(trainData, max_len=MAX_LEN)
    valDataset = myDataset(valData, max_len=MAX_LEN)
    train_loader = DataLoader(trainDataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(valDataset, batch_size=BATCH_SIZE)

    model = classifier(num_classes=4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    criterion = torch.nn.CrossEntropyLoss()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc='epoch:' + str(epoch)):
            input_ids = batch['input_ids'].to(device)
            # token_type_ids = batch['token_type_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            optimizer.zero_grad()
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
        train_loss = total_loss / len(train_loader)
        print(f'Epoch {epoch + 1}/{NUM_EPOCHS}')
        print(f'Train loss: {train_loss:.4f}')

        model.eval()
        total_loss = 0
        all_predictions = []
        all_labels = []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                # token_type_ids = batch['token_type_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                outputs = model(
                    input_ids=input_ids,
                    # token_type_ids=token_type_ids,
                    attention_mask=attention_mask,
                )
                loss = criterion(outputs, labels)
                total_loss += loss.item()
                predictions = torch.argmax(outputs, dim=1)

                all_predictions.extend(predictions.cpu().numpy().tolist())
                all_labels.extend(labels.cpu().numpy().tolist())
        val_loss = total_loss / len(val_loader)

        print(f'Epoch {epoch + 1}/{NUM_EPOCHS}')
        precision = metrics.precision_score(all_labels, all_predictions, average='micro',
                                            labels=[1, 2, 3])
        recall = metrics.recall_score(all_labels, all_predictions, average='micro',
                                      labels=[1, 2, 3])
        f1 = metrics.f1_score(all_labels, all_predictions, average='micro',
                              labels=[1, 2, 3])
        print('presion is %.4f, recall is %.4f, f1 score is %.4f\n' % (precision, recall, f1))
