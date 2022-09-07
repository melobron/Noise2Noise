from torch.utils.data import Dataset
import torchvision.transforms as transforms
from utils import *


class ImageNetGray(Dataset):
    def __init__(self, input_noise, target_noise, data_dir='../all_datasets/ImageNet_1000_Gray/', train=True, transform=None):
        super(ImageNetGray, self).__init__()

        self.input_type, self.input_intensity = input_noise.split('_')[0], float(input_noise.split('_')[1]) / 255.
        self.target_type, self.target_intensity = target_noise.split('_')[0], float(target_noise.split('_')[1]) / 255.

        if train:
            self.clean_dir = os.path.join(data_dir, 'train')
        else:
            self.clean_dir = os.path.join(data_dir, 'test')

        self.clean_paths = sorted(make_dataset(self.clean_dir))

        if transform:
            self.transform = transform
        else:
            self.transform = transforms.Compose([transforms.ToTensor()])

    def __getitem__(self, index):
        clean_path = self.clean_paths[index]
        clean = cv2.imread(clean_path, cv2.IMREAD_GRAYSCALE) / 255.

        # Input Noisy
        if self.input_type == 'gauss':
            input_noisy = clean + np.random.randn(*clean.shape) * self.input_intensity
        elif self.input_type == 'poisson':
            input_noisy = np.random.poisson(clean * 255. * self.input_intensity) / self.input_intensity / 255.
        else:
            raise NotImplementedError('wrong type of noise')

        # Target Noisy
        if self.target_type == 'gauss':
            target_noisy = clean + np.random.randn(*clean.shape) * self.target_intensity
        elif self.target_type == 'poisson':
            target_noisy = np.random.poisson(clean * 255. * self.target_intensity) / self.target_intensity / 255.
        else:
            raise NotImplementedError('wrong type of noise')

        clean, input_noisy, target_noisy = self.transform(clean), self.transform(input_noisy), self.transform(target_noisy)
        clean, input_noisy, target_noisy = clean.type(torch.FloatTensor), input_noisy.type(torch.FloatTensor), target_noisy.type(torch.FloatTensor)
        return {'clean': clean, 'input': input_noisy, 'target': target_noisy}

    def __len__(self):
        return len(self.clean_paths)










