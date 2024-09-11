def train(dataloader, testloader, model, epoch) -> None:
    size = len(dataloader.dataset)

    if (epoch == 1):
        model.file.write("0\t[ 0000/60000]\t--\t")  # Make sure that avg CE is the same as this.
        test(testloader, model)

    model.train()

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(DEVICE), y.to(DEVICE)

        # Compute prediction error
        pred = model(X)
        loss = model.criterion(pred, y)

        # Backpropagation
        model.optimizer.zero_grad()
        loss.backward()

        model.optimizer.step()

        pred = model(X)  # These two lines ensure that the
        loss = model.criterion(pred, y)  # Loss after the backwards pass is printed
        model.ces.append(loss.cpu().detach().numpy())

        loss, current = loss.item(), (batch + 1) * len(X)
        model.file.write(f"{epoch}\t[{current:>5d}/{size:>5d}]\t{loss:>7f}")

        test(testloader, model)
        if len(model.ces) >= 1000:
            model.stop_training = True
            break
    #For higher pruning percentages (35% and above) we are forcing all runs to B_max = 1000, and are removing the possibility of stopping earlier by removing the fitting function stop criteria
    #try:
    #params, inv_func = inv_fit(range(len(model.ces)), np.array(model.ces))
    #a, b, c = params
    #model.ce_asy_list.append(c)
    #model.a_list.append(a)
    #model.b_list.append(b)

    #if 0.95 * model.ce_asy_list[-2] < model.ce_asy_list[-1] < 1.05 * model.ce_asy_list[-2]:
    #model.stop_counter += 1
    #if model.stop_counter > 4:
    #print('stop training (fit)')
    #if len(model.ces) > 29:
    #model.stop_training = True
    #break
    #else:
    #model.stop_counter = 0

    #except (RuntimeError, TypeError) as error:
    #model.ce_asy_list.append(0)
    #model.a_list.append(0)
    #model.b_list.append(0)