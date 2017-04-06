import numpy as np
import os
import caffe

# Configuration
network_proto_path = ''
network_model_path = ''
data_mean = []
dataset_dir = 'Z:\\CASIA-WebFace-Clean-align'
visual = True

# Init Caffe Related
caffe.set_mode_gpu()
net = caffe.Net(network_proto_path, network_model_path, caffe.TEST)
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
if data_mean != None:
  transformer.set_mean('data', data_mean)
transformer.set_raw_scale('data', 255) # 256
transformer.set_transpose('data', (2,0,1))
transformer.set_channel_swap('data', (2,1,0))
batch_size = net.blobs['data'].data.shape[0]
print "Batch Size: ", batch_size
#print "Feature Layer Shape: ", net.blobs[layer_name].data.shape

# Extract Features based on images path list
def extract_features(image_list, layer_name='fc6'):    
    length = len(image_list)
    features = np.empty((length, net.blobs[layer_name].data.shape[1]), dtype='float32')
    img_batch = []
    for cnt, path in zip(range(length), image_list):
      im = transformer.preprocess('data', caffe.io.load_image(path))
      img_batch.append(im)
      if len(img_batch) == batch_size or cnt == (length-1):
        out = net.forward_all(data = np.array(img_batch), blobs=[layer_name])
        features[cnt-len(img_batch)+1:cnt+1, :] = np.float32(out[layer_name][0:len(img_batch),:].copy())
        img_batch = []
    return features

if __name__ == "__main__":
  for folder in os.listdir(dataset_dir):
    folder_dir = os.path.join(dataset_dir, folder)
    images_path = [os.path.join(folder_dir, item) for item in os.listdir(folder_dir)]
    features = extract_features(images_path)
    feature_center = np.median(features, axis=0)
    diff = np.subtract(np.array([feature_center]), features)
    l2dist = np.int32(np.sum(np.square(diff),1))
    # Visual L2 Distance
    if visual:
      import seaborn as sns
      import matplotlib.pyplot as plt
      fig = plt.figure()
      ax = fig.add_subplot(111)
      # the histogram of the data
      n, bins, patches = ax.hist(l2dist, len(l2dist)//2, facecolor='green')
      ax.set_xlabel('L2 Distance')
      ax.set_ylabel('Probability')
      ax.set_title(folder)
      ax.grid(True)
      plt.show()