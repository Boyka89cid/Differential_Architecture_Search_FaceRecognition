import os
import numpy as np
import random
import pandas as pd
from PIL import Image
from collections import Counter, defaultdict, namedtuple

from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from torchvision.transforms.functional import to_tensor

LFW_ROOT = os.path.expanduser(os.path.join("~", "Downloads", "lfw2"))
# LFW_ROOT = os.path.expanduser(os.path.join("~", "Downloads", "MTCNN", "lfw-deepfunneled-faces"))
LFW_PEOPLE = os.path.join(LFW_ROOT, "lfw-deepfunneled_crop-faces2")
# LFW_PEOPLE = os.path.join(LFW_ROOT, "people")
LFW_ANNOTATIONS = os.path.join(LFW_ROOT, "annotations")
LFW_TRAIN = os.path.join(LFW_ANNOTATIONS, "peopleDevTrain.txt")
LFW_TEST = os.path.join(LFW_ANNOTATIONS, "peopleDevTest.txt")
LFW_TRAIN_PAIRS = os.path.join(LFW_ANNOTATIONS, "pairsDevTrain.txt")
LFW_TEST_PAIRS = os.path.join(LFW_ANNOTATIONS, "pairsDevTest.txt")

# UMDSplit = namedtuple("UMDSplit", "split, annotations_file, people_directory")
# UMD_ROOT = os.path.join("/", "data", "umdFaces")
# UMD_ANNOTATIONS = os.path.join(UMD_ROOT, "annotations")
# UMD_SPLITS = {split : UMDSplit(split, "umdfaces_{}_ultraface.csv".format(split), "umdfaces_{0}".format(split))
#               for split in ["videos", "batch1", "batch2", "batch3"]}


ROOT_CELEB = os.path.expanduser(os.path.join("~", "Downloads", "image_align_celeb", "img_align_celeb"))
IMAGES_DIR = os.path.join(ROOT_CELEB, "img_align_celeba_png")
CELEB_ANNOTATIONS = os.path.join(ROOT_CELEB, "Anno")
IDENTITIES = os.path.join(CELEB_ANNOTATIONS, "identity_CelebA.txt")
PARTITIONS = os.path.join(CELEB_ANNOTATIONS, "list_eval_partition.txt")
TRAIN_FLAG = 0
VAL_FLAG = 1
TEST_FLAG = 2
IMAGE_CLASSES = pd.read_csv(IDENTITIES, sep=' ', header=None, names=['Image', 'Class'])
IMAGE_EVAL = pd.read_csv(PARTITIONS, sep=' ', header=None, names=['Image', 'Evaluation'])
IMAGE_CLASSES_EVAL = pd.merge(IMAGE_CLASSES, IMAGE_EVAL, on='Image')


class ImageNormalizationMeter:
    def __init__(self):
        self.meter = StandardScaler()

    def __call__(self, image_path):
        image = Image.open(image_path)
        image_tensor = to_tensor(image)
        pixel_stack = image_tensor.view(3, -1).t()
        self.meter.partial_fit(pixel_stack)
        return pixel_stack

    @property
    def mean(self):
        mean = getattr(self.meter, "mean_", None)
        return mean

    @property
    def std(self):
        var = getattr(self.meter, "var_", None)
        std = var if isinstance(var, type(None)) else np.sqrt(var)
        return std

    @property
    def parameters(self):
        return self.mean, self.std


class LFWClassificationDataset(Dataset):
    def __init__(self, train=True, pil=False, size=250, image_transform=None, target_transform=None):
        annotations_path = LFW_TRAIN if train else LFW_TEST
        self.pil = pil
        self.size = (size, size)  # TODO: IMPROVEMENT: Find the sweet spot for image/batch size~
        self.image_transform = image_transform
        self.target_transform = target_transform
        with open(annotations_path, 'r') as fp:
            self.image_classes = [entry.split()[0] for entry in fp.readlines()[1:]]
        self.images = []
        self.targets = []
        for image_class in self.image_classes:
            image_class_path = os.path.join(LFW_PEOPLE, image_class)
            for image_fn in os.listdir(image_class_path):
                image_path = os.path.join(image_class_path, image_fn)
                self.images.append(image_path)
                self.targets.append(image_class)
        self.images_pil = [None for _ in self.images]

    def __getitem__(self, item):
        image = self.images[item]
        if self.pil:
            if isinstance(self.images_pil[item], type(None)): self.images_pil[item] = Image.open(image).resize(
                self.size)
            image = self.images_pil[item]
        if not isinstance(self.image_transform, type(None)): image = self.image_transform(image)

        target = self.targets[item]
        if not isinstance(self.target_transform, type(None)): target = self.target_transform(target)
        return image, target

        # image = self.images[item]
        # target = self.targets[item]
        #
        # img = Image.open(image).resize(self.size)
        #
        # if self.image_transform is not None:
        #     img = self.image_transform(img)
        #
        # if self.target_transform is not None:
        #     target = self.target_transform(target)
        #
        # return img, target

    def __len__(self):
        return len(self.images)


class LFWPairsDataset(Dataset):

    def __init__(self, train=False, pil=False, size=250, image_transform=None, target_transform=None, root=LFW_ROOT):
        annotations_path = LFW_TRAIN_PAIRS if train else LFW_TEST_PAIRS
        self.pil = pil
        self.size = (size, size)
        self.image_transform = image_transform
        self.target_transform = target_transform
        self.pairs = []
        self.targets = []

        # self.people_path = os.path.join(os.path.expanduser(root), 'people')
        self.people_path = LFW_PEOPLE

        with open(annotations_path, 'r') as fp:
            annotation_lines = fp.readlines()
            n = int(annotation_lines[0].strip())
            for positive_pair_line in annotation_lines[1:n + 1]:
                pair, target = self._parse_positive_pair(positive_pair_line)
                self.pairs.append(pair)
                self.targets.append(target)
            for negative_pair_line in annotation_lines[n + 1:]:
                pair, target = self._parse_negative_pair(negative_pair_line)
                self.pairs.append(pair)
                self.targets.append(target)
        self.images_pil = {}

    def _parse_positive_pair(self, positive_pair_line):
        target = True
        person, entry_a, entry_b = positive_pair_line.strip().split()
        person_entry_a_path = self._get_person_entry_path(person, int(entry_a))
        person_entry_b_path = self._get_person_entry_path(person, int(entry_b))
        return (person_entry_a_path, person_entry_b_path), target

    def _parse_negative_pair(self, negative_pair_line):
        target = False
        person_a, entry_a, person_b, entry_b = negative_pair_line.strip().split()
        person_entry_a_path = self._get_person_entry_path(person_a, int(entry_a))
        person_entry_b_path = self._get_person_entry_path(person_b, int(entry_b))
        return (person_entry_a_path, person_entry_b_path), target

    def _get_person_entry_path(self, person, entry):
        entry_fn = "{person}_{entry:0>4d}.jpg".format(person=person, entry=entry)
        entry_path = os.path.join(self.people_path, person, entry_fn)
        return entry_path

    def __getitem__(self, item):
        pair, target = self.pairs[item], self.targets[item]
        image_a, image_b = pair
        if self.pil:
            if isinstance(self.images_pil.get(image_a, None), type(None)): self.images_pil[image_a] = Image.open(
                image_a).resize(self.size)
            if isinstance(self.images_pil.get(image_b, None), type(None)): self.images_pil[image_b] = Image.open(
                image_b).resize(self.size)
            image_a = self.images_pil[image_a]
            image_b = self.images_pil[image_b]
        if not isinstance(self.image_transform, type(None)):
            image_a = self.image_transform(image_a)
            image_b = self.image_transform(image_b)
        if not isinstance(self.target_transform, type(None)): target = self.target_transform(target)
        pair = (image_a, image_b)
        return pair, target

    def __len__(self):
        return len(self.pairs)


class CelebADataset(Dataset):
    def __init__(self, flag=0, pil=False, size=250, image_transform=None, target_transform=None, num_img=None):
        self.pil = pil
        self.image_transform = image_transform
        self.target_transform = target_transform
        self.flag = flag
        self.size = (size, size)
        self.num_img = num_img
        filtered_df = IMAGE_CLASSES_EVAL.loc[IMAGE_CLASSES_EVAL['Evaluation'] == self.flag]
        filtered_df = filtered_df.reset_index(drop=True)
        self.imgs = []
        self.targets = []

        filtered_df['Image'] = filtered_df['Image'].map(lambda x: os.path.join(IMAGES_DIR, str(x)[:-3] + 'png'))

        if self.num_img is not None:
            filtered_df = filtered_df.groupby('Class', group_keys=False).apply(
                lambda x: x.sample(min(len(x), self.num_img)))
            filtered_df = filtered_df.reset_index(drop=True)

        self.imgs = list(filtered_df['Image'].values)
        self.targets = list(filtered_df['Class'].values)

        # self.images_pil = [None for _ in self.imgs]

    def __getitem__(self, item):
        image = self.imgs[item]
        # if self.pil:
        #     if isinstance(self.images_pil[item], type(None)): self.images_pil[item] = Image.open(image).
        #     resize(self.size)
        #     image = self.images_pil[item]
        image = Image.open(image).resize(self.size)
        if not isinstance(self.image_transform, type(None)): image = self.image_transform(image)
        target = self.targets[item]
        if not isinstance(self.target_transform, type(None)): target = self.target_transform(target)
        return image, target

    def __len__(self):
        return len(self.imgs)


class TripletDataset(Dataset):
    """
    "If I have seen further it is only by standing on the shoulders of giants."
    https://github.com/adambielski/siamese-triplet/blob/torch-0.3.1/datasets.py
    """

    def __init__(self, classification_dataset, fix_triplets=False):
        self.classification_dataset = classification_dataset
        self.fix_triplets = fix_triplets
        self.image_classes = set(self.classification_dataset.image_classes)
        self.image_class_counts = Counter(e[1] for e in self.classification_dataset)
        self.anchor_image_classes = set(image_class
                                        for image_class in self.image_class_counts.keys()
                                        if self.image_class_counts[image_class] > 1)
        self.image_classes_to_dataset_indices = defaultdict(list)
        for i, (image, image_class) in enumerate(self.classification_dataset):
            self.image_classes_to_dataset_indices[image_class].append(i)
        self.step = 0
        self.set_triplets()

    def set_triplets(self):
        if self.fix_triplets: random.seed(42)
        classes = []
        anchors = []
        positives = []
        negatives = []
        for anchor_image_class in self.anchor_image_classes:
            anchor_dataset_indices = self.image_classes_to_dataset_indices[anchor_image_class]
            anchors.extend(anchor_dataset_indices)
            classes.extend([anchor_image_class for _ in anchor_dataset_indices])
        for anchor_index, anchor in enumerate(anchors):
            anchor_image_class = classes[anchor_index]
            negative_image_class = random.choice(list(self.image_classes - {anchor_image_class, }))
            positives.append(random.choice(self.image_classes_to_dataset_indices[anchor_image_class]))
            negatives.append(random.choice(self.image_classes_to_dataset_indices[negative_image_class]))
        self.triplets = list(zip(anchors, positives, negatives, classes))

    def __getitem__(self, item):
        self.step = (self.step + 1) % len(self)
        if not self.step: self.set_triplets()  # TODO: IMPROVEMENT: Can this be sped up?
        return tuple(self.classification_dataset[i][0] for i in self.triplets[item][:-1])

    def __len__(self):
        return len(self.triplets)

# if __name__ == '__main__':
# image_normalization_meter = ImageNormalizationMeter()
# lfw_classification_dataset_train = LFWClassificationDataset(image_transform=image_normalization_meter, train=True)
# for image in lfw_classification_dataset_train: pass
# lfw_classification_dataset_test = LFWClassificationDataset(image_transform=image_normalization_meter, train=False)
# for image in lfw_classification_dataset_test: pass
# print(image_normalization_meter.parameters)
# result: (array([0.43920362, 0.38309247, 0.34243814]), array([0.29700514, 0.27357439, 0.26827176]))
# umdfaces_classification_dataset = UMDFacesClassificationDataset(split='batch1', pil=True)
