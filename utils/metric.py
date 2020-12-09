
def test(net, test_loader):
    net.eval()

    correct = 0

    for i, (data, target) in enumerate(test_loader):
        data, target = data.cuda(), target.cuda()
        output = net(data)

        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()


    acc = correct / len(test_loader.dataset)
    return acc
    