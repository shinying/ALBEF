import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image

from dataset.caption_dataset import re_train_dataset, re_eval_dataset, pretrain_dataset
from dataset.nlvr_dataset import nlvr_dataset
from dataset.ve_dataset import ve_dataset
from dataset.vqa_dataset import vqa_dataset
from dataset.vlep_dataset import vlep_dataset # mine
from dataset.nextqa_dataset import nextqa_dataset
from dataset.grounding_dataset import grounding_dataset

from dataset.randaugment import RandomAugment

def create_dataset(dataset, config):

    normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))

    pretrain_transform = transforms.Compose([
            transforms.RandomResizedCrop(config['image_res'],scale=(0.2, 1.0), interpolation=Image.BICUBIC),
            transforms.RandomHorizontalFlip(),
            RandomAugment(2,7,isPIL=True,augs=['Identity','AutoContrast','Equalize','Brightness','Sharpness',
                                              'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),
            transforms.ToTensor(),
            normalize,
        ])
    train_transform = transforms.Compose([
            transforms.RandomResizedCrop(config['image_res'],scale=(0.5, 1.0), interpolation=Image.BICUBIC),
            transforms.RandomHorizontalFlip(),
            RandomAugment(2,7,isPIL=True,augs=['Identity','AutoContrast','Equalize','Brightness','Sharpness',
                                              'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),
            transforms.ToTensor(),
            normalize,
        ])
    test_transform = transforms.Compose([
        transforms.Resize((config['image_res'],config['image_res']),interpolation=Image.BICUBIC),
        transforms.ToTensor(),
        normalize,
        ])

    if dataset=='pretrain':
        dataset = pretrain_dataset(config['train_file'], pretrain_transform)
        return dataset

    elif dataset=='re':
        train_dataset = re_train_dataset(config['train_file'], train_transform, config['image_root'])
        val_dataset = re_eval_dataset(config['val_file'], test_transform, config['image_root'])
        test_dataset = re_eval_dataset(config['test_file'], test_transform, config['image_root'])
        return train_dataset, val_dataset, test_dataset

    elif dataset=='vqa':
        train_dataset = vqa_dataset(config['train_file'], train_transform, config['vqa_root'], config['vg_root'], split='train')
        vqa_test_dataset = vqa_dataset(config['test_file'], test_transform, config['vqa_root'], config['vg_root'], split='test', answer_list=config['answer_list'])
        return train_dataset, vqa_test_dataset

    elif dataset=='nlvr':
        train_dataset = nlvr_dataset(config['train_file'], train_transform, config['image_root'])
        val_dataset = nlvr_dataset(config['val_file'], test_transform, config['image_root'])
        test_dataset = nlvr_dataset(config['test_file'], test_transform, config['image_root'])
        return train_dataset, val_dataset, test_dataset

    elif dataset=='ve':
        train_dataset = ve_dataset(config['train_file'], train_transform, config['image_root'])
        val_dataset = ve_dataset(config['val_file'], test_transform, config['image_root'])
        test_dataset = ve_dataset(config['test_file'], test_transform, config['image_root'])
        return train_dataset, val_dataset, test_dataset

    elif dataset=='grounding':
        train_transform = transforms.Compose([
                transforms.Resize((config['image_res'],config['image_res']),interpolation=Image.BICUBIC),
                transforms.RandomHorizontalFlip(),
                RandomAugment(2,7,isPIL=True,augs=['Identity','AutoContrast','Equalize','Brightness','Sharpness',
                                                  'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),
                transforms.ToTensor(),
                normalize,
            ])
        train_dataset = grounding_dataset(config['train_file'], train_transform, config['image_root'], mode='train')
        test_dataset = grounding_dataset(config['test_file'], test_transform, config['image_root'], mode='test')
        return train_dataset, test_dataset

    elif dataset=='vlep':
        train_dataset = vlep_dataset(config['root'], 'train', train_transform)
        val_dataset = vlep_dataset(config['root'], 'dev', test_transform)
        return train_dataset, val_dataset

    elif dataset == 'nextqa':
        train_dataset = nextqa_dataset(config['root'], 'train', config['nframe'], train_transform)
        val_dataset = nextqa_dataset(config['root'], 'val', config['nframe'],test_transform)
        test_dataset = nextqa_dataset(config['root'], 'test', config['nframe'],test_transform)
        return train_dataset, val_dataset, test_dataset


def vqa_collate_fn(batch):
    image_list, question_list, answer_list, weight_list, n = [], [], [], [], []
    for image, question, answer, weights in batch:
        image_list.append(image)
        question_list.append(question)
        weight_list += weights
        answer_list += answer
        n.append(len(answer))
    return torch.stack(image_list,dim=0), question_list, answer_list, torch.Tensor(weight_list), n


def vlep_collate_fn(batch):
    frames = torch.stack([data[0] for data in batch])
    B, C, H, W = frames.size()
    frames = frames.unsqueeze(1).repeat(1, 2, 1, 1, 1).view(B*2, C, H, W)
    texts = [event for data in batch for event in data[1]]
    answers = torch.tensor([data[2] for data in batch], dtype=torch.long)
    return frames, texts, answers


def nextqa_collate_fn(batch):
    frames = torch.stack([d for data in batch for d in data[0]])
    questions = [data[1] for data in batch]
    choices = [choice for data in batch for choice in data[2]]
    labels = torch.tensor([data[3] for data in batch], dtype=torch.long)
    nframe = [data[4] for data in batch]
    return frames, questions, choices, labels, nframe


def create_sampler(datasets, shuffles, num_tasks, global_rank):
    samplers = []
    for dataset,shuffle in zip(datasets,shuffles):
        sampler = torch.utils.data.DistributedSampler(dataset, num_replicas=num_tasks, rank=global_rank, shuffle=shuffle)
        samplers.append(sampler)
    return samplers


def create_loader(datasets, samplers, batch_size, num_workers, is_trains, collate_fns):
    loaders = []
    for dataset,sampler,bs,n_worker,is_train,collate_fn in zip(datasets,samplers,batch_size,num_workers,is_trains,collate_fns):
        if is_train:
            shuffle = (sampler is None)
            drop_last = True
        else:
            shuffle = False
            drop_last = False
        loader = DataLoader(
            dataset,
            batch_size=bs,
            num_workers=n_worker,
            pin_memory=True,
            sampler=sampler,
            shuffle=shuffle,
            collate_fn=collate_fn,
            drop_last=drop_last,
        )
        loaders.append(loader)
    return loaders
