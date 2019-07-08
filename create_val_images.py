from skimage import io
import torch
import matplotlib.pyplot as plt
import os

VALS = {
        'data_path_root': '/qfs/projects/sgdatasc/spacenet/',
        'data_path_test': 'Vegas_processed_test/annotations'
       }

data_path_test = os.path.join(data_path_root, VALS['data_path_test'])
dset_test = SpaceNetDataset(data_path_test, split_tags, transform=T.Compose([ToTensor()]))
loader_test = DataLoader(dset_test, batch_size=test_batch_size, shuffle=True,
                         num_workers=num_workers)

image = io.imread(image_path)
image_tensor = image.transpose((2, 0, 1))
image_tensor = image_tensor.reshape((1, image_tensor.shape[0], image_tensor.shape[1], image_tensor.shape[2]))
image_tensor = torch.from_numpy(image_tensor).type(torch.float32)
print(image_tensor.shape)

model.eval()  # set model to evaluation mode
with torch.no_grad():
    scores = model(image_tensor)
    _, prediction = scores.max(1)

prediction = prediction.transpose((1, 2, 0))
fig = plt.figure()
plt.imshow(image, interpolation='none')
plt.imshow(prediction, cmap='grey', interpolation='none', alpha=0.5)

if save_to_dir is not None:
    fig.savefig(os.path.join(save_to_dir, result_image_name))
