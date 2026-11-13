import numpy as np

'''
default setting : non-iid & dirchlet_parameter is 0.5
Implementation follows standard non-IID data partitioning using Dirichlet distribution.
'''
def noniid(dataset, args):

    idxs = np.arange(len(dataset))

    if args.dataset in ["CIFAR10", "MNIST", "FMNIST"]:

        labels = np.transpose(np.array(dataset.targets))
    else:
        labels = np.transpose(np.array(dataset.labels))


    dict_users = {i: list() for i in range(args.num_clients)}
    dict_labels = dict()
    
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:,idxs_labels[1,:].argsort()]
    
    idxs = list(idxs_labels[0])
    labels = idxs_labels[1]
     
    min_num, max_num = 500, 1500
    if args.num_clients > 50:
        min_num, max_num = 250, 700
    num_rand = np.random.randint(min_num, max_num+1, size=args.num_clients)

    proportions = np.random.dirichlet(np.ones(args.num_classes) * 0.5, args.num_clients)
        
    for i in range(args.num_classes):
        specific_class = set(np.extract(labels == i, idxs))
        dict_labels.update({i : specific_class})

    for i, prop in enumerate(proportions):
        prop = num_rand[i] * prop
                
        rand_set = list()
        for c in range(args.num_classes):
            try:
                rand_class = list(np.random.choice(list(dict_labels[c]), int(prop[c])))
                dict_labels[c] = dict_labels[c] - set(rand_class)
                rand_set = rand_set + rand_class
            except ValueError:
                pass
        dict_users[i] = set(rand_set)

    return dict_users