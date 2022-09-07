import argparse
import random
import time
from glob import glob

from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

from models.DnCNN import DnCNN
from utils import *

# Arguments
parser = argparse.ArgumentParser(description='Test N2N public')

parser.add_argument('--gpu_num', default=0, type=int)
parser.add_argument('--seed', default=100, type=int)
parser.add_argument('--exp_num', default=1, type=int)

# Model parameters
parser.add_argument('--n_epochs', default=200, type=int)

# Test parameters
parser.add_argument('--noise', default='poisson_50', type=str)  # 'gauss_intensity', 'poisson_intensity'
parser.add_argument('--dataset', default='BSD100', type=str)  # BSD100, Kodak, Set12

# Transformations
parser.add_argument('--crop', type=bool, default=True)
parser.add_argument('--patch_size', type=int, default=256)
parser.add_argument('--normalize', type=bool, default=True)
parser.add_argument('--mean', type=float, default=0.4050)  # ImageNet Gray: 0.4050
parser.add_argument('--std', type=float, default=0.2927)  # ImageNet Gray: 0.2927

opt = parser.parse_args()


def generate(args):
    device = torch.device('cuda:{}'.format(args.gpu_num))

    # Random Seeds
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Model
    model = DnCNN().to(device)
    model.load_state_dict(torch.load('./experiments/exp{}/checkpoints/{}epochs.pth'.format(args.exp_num, args.n_epochs), map_location=device))
    model.eval()

    # Directory
    img_dir = os.path.join('../all_datasets/', args.dataset)
    save_dir = os.path.join('./results/', args.dataset)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    # Images
    img_paths = glob(os.path.join(img_dir, '*.png'))
    imgs = [cv2.imread(p, cv2.IMREAD_GRAYSCALE) for p in img_paths]

    # Noise
    noise_type = args.noise.split('_')[0]
    noise_intensity = int(args.noise.split('_')[1]) / 255.

    # Transform
    transform = transforms.Compose(get_transforms(args))

    # Denoising
    noisy_psnr, denoised_psnr = 0, 0
    noisy_ssim, denoised_ssim = 0, 0

    avg_time = 0

    for index, clean255 in enumerate(imgs):
        if args.crop:
            clean255 = crop(clean255, patch_size=args.patch_size)

        clean_numpy = clean255/255.
        if noise_type == 'gauss':
            noisy_numpy = clean_numpy + np.random.normal(size=clean_numpy.shape) * noise_intensity
        elif noise_type == 'poisson':
            noisy_numpy = np.random.poisson(clean_numpy * 255. * noise_intensity) / noise_intensity / 255.
        else:
            raise NotImplementedError('Wrong Noise')

        noisy = transform(noisy_numpy)
        noisy = torch.unsqueeze(noisy, dim=0)
        noisy = noisy.type(torch.FloatTensor).to(device)

        start = time.time()
        denoised = model(noisy)
        elapsed = time.time() - start
        avg_time += elapsed / len(imgs)

        # Change to Numpy
        if args.normalize:
            denoised = denorm(denoised, mean=args.mean, std=args.std)
        denoised = tensor_to_numpy(denoised)
        denoised_numpy = np.squeeze(denoised)

        # Calculate PSNR
        n_psnr = psnr(clean_numpy, noisy_numpy, data_range=1)
        d_psnr = psnr(clean_numpy, denoised_numpy, data_range=1)

        noisy_psnr += n_psnr / len(imgs)
        denoised_psnr += d_psnr / len(imgs)

        # Calculate SSIM
        n_ssim = ssim(clean_numpy, noisy_numpy, data_range=1)
        d_ssim = ssim(clean_numpy, denoised_numpy, data_range=1)

        noisy_ssim += n_ssim / len(imgs)
        denoised_ssim += d_ssim / len(imgs)

        print('{}th image | PSNR: noisy:{:.3f}, denoised:{:.3f} | SSIM: noisy:{:.3f}, denoised:{:.3f}'.format(
            index+1, n_psnr, d_psnr, n_ssim, d_ssim))

        # Save sample images
        if index <= 3:
            sample_clean, sample_noisy = 255. * np.clip(clean_numpy, 0., 1.), 255. * np.clip(noisy_numpy, 0., 1.)
            sample_denoised = 255. * np.clip(denoised_numpy, 0., 1.)
            cv2.imwrite(os.path.join(save_dir, '{}th_clean.png'.format(index+1)), sample_clean)
            cv2.imwrite(os.path.join(save_dir, '{}th_noisy.png'.format(index+1)), sample_noisy)
            cv2.imwrite(os.path.join(save_dir, '{}th_denoised.png'.format(index+1)), sample_denoised)

    # Total PSNR, SSIM
    print('{} Average PSNR | noisy:{:.3f}, denoised:{:.3f}'.format(args.dataset, noisy_psnr, denoised_psnr))
    print('{} Average SSIM | noisy:{:.3f}, denoised:{:.3f}'.format(args.dataset, noisy_ssim, denoised_ssim))
    print('Average Time for Denoising | denoised:{}'.format(avg_time))


if __name__ == "__main__":
    generate(opt)
