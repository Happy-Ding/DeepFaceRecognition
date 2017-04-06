import numpy as np
import os
import caffe

# Configuration
network_proto_path = 'Models/deploy.prototxt'
network_model_path = 'Models/face_model.caffemodel'
data_mean = np.array([127.5, 127.5, 127.5])
dataset_dir = 'Z:\\CASIA-WebFace-Clean-align' #msceleb-align
visual = False

# Init Caffe Related
caffe.set_mode_gpu()
net = caffe.Net(network_proto_path, network_model_path, caffe.TEST)
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
if data_mean is not None:
  transformer.set_mean('data', data_mean)
transformer.set_raw_scale('data', 255) # 256
transformer.set_transpose('data', (2,0,1))
transformer.set_channel_swap('data', (2,1,0))
batch_size = net.blobs['data'].data.shape[0]
print("Batch Size: "+str(batch_size))
#print "Feature Layer Shape: ", net.blobs[layer_name].data.shape

# Extract Features based on images path list
def extract_features(image_list, layer_name='fc5'):    
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
  records = []
  number_total = 0
  for index, folder in enumerate(os.listdir(dataset_dir)):
    folder_dir = os.path.join(dataset_dir, folder)
    images_path = np.array([os.path.join(folder_dir, item) for item in os.listdir(folder_dir)])
    features = extract_features(images_path)
    feature_center = np.median(features, axis=0)
    diff = np.subtract(np.array([feature_center]), features)
    l2dist = np.int32(np.sum(np.square(diff),1))
    for item in images_path[l2dist <= 650]:
      records.append((item, index))
    number_total += images_path.shape[0]
    # Visual L2 Distance
    if visual:
      # Python3 SeaBorn Bug FIX: 
      # conda uninstall statsmodels --yes
      # conda install -c taugspurger statsmodels=0.8.0
      import seaborn as sns
      import matplotlib.pyplot as plt
      ax = sns.kdeplot(l2dist, shade=True)
      ax.set_xlabel('L2 Distance')
      ax.set_ylabel('Probability')
      ax.set_title(folder_dir)
      plt.show()
  print("Total: "+str(number_total))
  print("Output: "+str(len(records)))
  with open("list.txt", 'w') as f:
    f.write('\n'.join([path+' '+str(label) for path, label in records]))