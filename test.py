import time
import os
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
import torchvision.utils as vutils
import time

opt = TestOptions().parse()
opt.nThreads = 1   # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
model = create_model(opt)

print(len(dataset))
model = model.eval()
print(model.training)

opt.how_many = 999999
# test
dest_root = os.path.join(opt.checkpoints_dir, opt.name, opt.which_epoch)
if not os.path.exists(dest_root):
    os.makedirs(dest_root)
    
for i, data in enumerate(dataset):
    print(' process %d/%d img ..'%(i,opt.how_many))
    if i >= opt.how_many:
        break
    model.set_input(data)
    startTime = time.time()
    model.test()
    endTime = time.time()
    print(endTime-startTime)
    visuals = model.get_current_visuals_save()
    new_path = os.path.join(dest_root, str(i) + '.jpg')
    vutils.save_image(visuals, new_path, normalize=True)



