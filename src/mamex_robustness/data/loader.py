import os
from collections import defaultdict
import numpy as np
import scipy.sparse as sp
import torch
from tqdm import tqdm

from mamex_robustness.domain.noise import CorruptionSpec, TextNoiseSimulator


class Loader4MM(torch.utils.data.Dataset):
    def __init__(self, env):

        self.env = env

        self.n_user = 0
        self.m_item = 0

        train_file = os.path.join(self.env.DATA_PATH, 'train.txt')
        val_file = os.path.join(self.env.DATA_PATH, 'val.txt')
        test_file = os.path.join(self.env.DATA_PATH, 'test.txt')

        trainUniqueUsers, trainItem, trainUser =[], [], []
        valUniqueUsers, valItem, valUser =[], [], []
        testUniqueUsers, testItem, testUser =[], [], []

        self.traindataSize = 0
        self.testDataSize = 0

        self.train_data = defaultdict(list)
        with open(train_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n').split(' ')
                    if l[1]=='':
                        continue
                    items = [int(i) for i in l[1:]]
                    uid = int(l[0])
                    self.train_data[uid].extend(items)
                    trainUniqueUsers.append(uid)
                    trainUser.extend([uid] * len(items))
                    trainItem.extend(items)
                    self.m_item = max(self.m_item, max(items))
                    self.n_user = max(self.n_user, uid)
                    self.traindataSize += len(items)
        self.trainUniqueUsers = np.array(trainUniqueUsers)
        self.trainUser = np.array(trainUser)
        self.trainItem = np.array(trainItem)
        self.trainItem = set(self.trainItem)

        self.cold_item_index = set()
        self.val_data = defaultdict(list)
        with open(val_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n').split(' ')
                    uid = int(l[0])
                    self.n_user = max(self.n_user, uid)
                    if l[1] == '':
                        continue
                    else:
                        items = [int(i) for i in l[1:]]

                        for item in items:
                            if item not in self.trainItem:
                                self.cold_item_index.add(item)
                    self.val_data[uid].extend(items)
                    valUniqueUsers.append(uid)
                    valUser.extend([uid] * len(items))
                    valItem.extend(items)
                    self.m_item = max(self.m_item, max(items))
                    # self.valDataSize += len(items)
        self.val_user_list = np.array(valUniqueUsers)

        self.test_data = defaultdict(list)
        with open(test_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n').split(' ')
                    uid = int(l[0])
                    self.n_user = max(self.n_user, uid)
                    if l[1] == '':
                        continue
                    else:
                        items = [int(i) for i in l[1:]]
                        for item in items:
                            if item not in self.trainItem:
                                self.cold_item_index.add(item)
                    self.test_data[uid].extend(items)
                    testUniqueUsers.append(uid)
                    testUser.extend([uid] * len(items))
                    testItem.extend(items)
                    self.m_item = max(self.m_item, max(items))
                    self.testDataSize += len(items)
        self.cold_item_index = list(self.cold_item_index)
        self.m_item += 1
        self.n_user += 1

        self.test_user_list = np.array(testUniqueUsers)
        self.testUser = np.array(testUser)
        self.testItem = np.array(testItem)


        # pre-calculate
        self._allPos = self.getUserAllItems()

        all_item_index = np.array(list(range(self.m_item)))
        self.cold_item_index = np.load(os.path.join(self.env.DATA_PATH, 'cold_item_index.npy'))
        self.warm_item_index = np.setdiff1d(all_item_index, self.cold_item_index, assume_unique=False)

        # optional ablation: warm items whose training interactions are removed (they stay in the item set, so they appear as
        # negatives like the cold items do); allPos and the validation and test data are unchanged
        self.heldout_warm_index = np.array([], dtype=int)
        holdout = float(getattr(self.env.args, 'holdout_warm', 0.0))
        if holdout > 0:
            rng = np.random.default_rng(int(getattr(self.env.args, 'holdout_seed', 0)))
            count = int(round(holdout * len(self.warm_item_index)))
            self.heldout_warm_index = np.sort(rng.choice(self.warm_item_index, size=count, replace=False))
            dropped = set(self.heldout_warm_index.tolist())
            for user in list(self.train_data.keys()):
                self.train_data[user] = [i for i in self.train_data[user] if i not in dropped]

        self.image_feat, self.text_feat, self.audio_feat = self.load_mutimedia_feature()
        self.feature = np.concatenate([self.image_feat, self.text_feat], axis=1)
        if self.env.args.dataset == 'tiktok':
            self.feature = np.concatenate([self.feature, self.audio_feat], axis=1)
            


    @property
    def allPos(self):
        return self._allPos

    def load_mutimedia_feature(self):
 
        image_file = os.path.join(self.env.DATA_PATH, 'image_feat.npy')
        text_file = os.path.join(self.env.DATA_PATH, 'text_feat.npy')

        # low_dim_feature_file = os.path.join(self.env.DATA_PATH, 'low_dim_feature.npy')
        image_feat = np.load(image_file)
        text_feat = np.load(text_file)
        if self.env.args.dataset == 'tiktok':
            audio_feat = np.load(audio_file)
        # low_dim_feature = np.load(low_dim_feature_file)
        if self.env.args.dataset == 'tiktok':
            return image_feat, text_feat, audio_feat
        else:
            return image_feat, text_feat, None


    def noisy_text_feat(self, noise_rate, noise_type='hashtag', seed=2025):
        """Corrupted copy of text_feat; only cold items are touched, self.text_feat is unchanged."""
        simulator = TextNoiseSimulator(CorruptionSpec(noise_rate, noise_type, seed))
        return simulator(self.text_feat, self.cold_item_index, self.warm_item_index)

    def train_noisy_text_feat(self, noise_rate, noise_type='hashtag', seed=2025):
        """Corrupted copy of text_feat for noise-augmented training: only warm items are touched, because cold items are never seen in training."""
        simulator = TextNoiseSimulator(CorruptionSpec(noise_rate, noise_type, seed))
        return simulator(self.text_feat, self.warm_item_index, self.warm_item_index)

    def getUserAllItems(self):
        """
        Get the items each user interacted with in the training and validation sets
        Used for negative sampling
        """
        posItems = defaultdict(list)

        for user in list(self.train_data.keys()):
            posItems[user].extend(self.train_data[user])
        return posItems

    def neg_sample(self):
        self.pair_data = []
        print('generate samples ...')
        for user_id in tqdm(self.train_data.keys()):
            positive_list = self.train_data[user_id]  # self.train_dict[user_id]
            for item_i in positive_list:
                item_j = np.random.randint(self.m_item)
                while item_j in positive_list:
                    item_j = np.random.randint(self.m_item)
                self.pair_data.append([user_id, item_i, item_j])

    def neg_uniform_sample(self):
        user_num = len(self.trainUser)
        users = np.random.randint(0, self.n_user, user_num)
        self.pair_data = []
        print('generate uniform samples ...')
        for user_id in tqdm(users):
            positive_list = self.train_data[user_id]  # self.train_dict[user_id]
            if len(positive_list) == 0:
                continue
            posindex = np.random.randint(0, len(positive_list))
            item_i = positive_list[posindex]
            item_j = np.random.randint(self.m_item)
            while item_j in positive_list:
                item_j = np.random.randint(self.m_item)
            self.pair_data.append([user_id, item_i, item_j])

    def __getitem__(self, index):
        user = self.pair_data[index][0]
        pos_item = self.pair_data[index][1]
        neg_item = self.pair_data[index][2]
        return user, pos_item, neg_item

    def __len__(self):
        return len(self.trainItem)

def Uniform_PairSample(dataset):
    """
    the original impliment of BPR Sampling in LightGCN
    :return:
        np.array
    """
    user_num = len(dataset.trainItem)
    users = np.random.randint(0, dataset.n_user, user_num)
    allPos = dataset.allPos
    S = []
    for i, user in enumerate(users):
        posForUser = allPos[user]
        if len(posForUser) == 0:
            continue
        posindex = np.random.randint(0, len(posForUser))
        positem = posForUser[posindex]
        while True:
            negitem = np.random.randint(0, dataset.m_item)
            if negitem in posForUser:
                continue
            else:
                break
        S.append([user, positem, negitem])
    return np.array(S)

def PairSample(dataset):
    """
    the original impliment of BPR Sampling in LightGCN
    :return:
        np.array
    """
    allPos = dataset.allPos
    S = []
    for i, user in enumerate(dataset.train_data.keys()):
        posForUser = allPos[user]
        if len(posForUser) == 0:
            continue
        for positem in dataset.train_data[user]:
            while True:
                negitem = np.random.randint(0, dataset.m_item)
                if negitem in posForUser:
                    continue
                else:
                    break
            random_user = np.random.randint(0, dataset.n_user)
            random_item = np.random.randint(0, dataset.m_item)
            S.append([user, positem, negitem, random_user, random_item])
    return np.array(S)

def minibatch(*tensors, batch_size):
    if len(tensors) == 1:
        tensor = tensors[0]
        for i in range(0, len(tensor), batch_size):
            yield tensor[i:i + batch_size]
    else:
        for i in range(0, len(tensors[0]), batch_size):
            yield tuple(x[i:i + batch_size] for x in tensors)