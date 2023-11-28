from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from pycocotools.coco import COCO
import os
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances_argmin
import numpy as np
from sklearn.decomposition import PCA
import torch
from sklearn.metrics import silhouette_score

"""
Maybe run this function only once at the beginning of the first batch
"""
# Determine the optimal k and return the model initialized with it
def get_kmeans_with_optimal_k(data, max_k):
    silhoutte_scores = [0] * (max_k)
    silhoutte_scores[0] = 1e10
    for k in range(2, max_k+1):
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(data)
        labels = kmeans.labels_
        silhoutte_scores.append(silhouette_score(data, labels))
    
    optimal_k = np.argmin(silhoutte_scores) + 1
    # print(optimal_k)
    final_kmeans = KMeans(n_clusters=optimal_k)
    final_kmeans.fit(data)

    return final_kmeans

def kmeans_centroids(image_feat, text_feat, max_k=20):
    # image_feat_2d = PCA(n_components=2).fit_transform(image_feat.cpu().numpy())
    # text_feat_2d = PCA(n_components=2).fit_transform(text_feat.cpu().numpy())
    # image_feat_std = StandardScaler().fit_transform(image_feat_2d)
    # text_feat_std = StandardScaler().fit_transform(text_feat_2d)

    image_feat_std = StandardScaler().fit_transform(image_feat.cpu().numpy())
    text_feat_std = StandardScaler().fit_transform(text_feat.cpu().numpy())

    # print(len(image_feat_std))

    # kmean_image = KMeans(n_clusters=k).fit(image_feat_std)
    # kmean_text = KMeans(n_clusters=k).fit(text_feat_std)
    kmean_image = get_kmeans_with_optimal_k(image_feat_std, max_k)
    kmean_text = get_kmeans_with_optimal_k(text_feat_std, max_k)

    image_centroids = kmean_image.cluster_centers_
    text_centroids = kmean_text.cluster_centers_

    image_cluster_idxs = pairwise_distances_argmin(image_feat_std, image_centroids)
    text_cluster_idxs = pairwise_distances_argmin(text_feat_std, text_centroids)

    image_centroids = torch.from_numpy(image_centroids).to(image_feat.device)
    text_centroids = torch.from_numpy(text_centroids).to(text_feat.device)
    image_cluster_idxs = torch.from_numpy(image_cluster_idxs).to(image_feat.device)
    text_cluster_idxs = torch.from_numpy(text_cluster_idxs).to(text_feat.device)

    return image_centroids, text_centroids,image_cluster_idxs, text_cluster_idxs


if __name__ == '__main__':
    image_feats = np.random.rand(20, 2048)
    text_feats = np.random.rand(20, 768)
    image_centroids, image_cluster_idxs, text_centroids, text_cluster_idxs = kmeans_centroids(image_feats, text_feats, 4)
    print(image_centroids)
    print(image_cluster_idxs)
    print(text_centroids)
    print(text_cluster_idxs)
    # dataDir = './mscoco_val'
    # dataType = 'mscoco_val2014_subset_5k'
    # annotationsFile = './mscoco_val/annotations/captions_val2014.json'

    # # storing currently obatined image IDs
    # images_path = os.path.join(dataDir, dataType)
    # # image_files = os.listdir(images_path)

    # # image_ids_5k = []
    # # for filename in image_files:
    # #     id_str = filename.split('_')[-1].split('.')[0]
    # #     image_id = int(id_str.lstrip('0'))
    # #     image_ids_5k.append(str(image_id))

    # # with open('image_ids_5k.json', 'w') as file:
    # #     json.dump(image_ids_5k, file)

    # coco = COCO(annotationsFile)

    # # img_ids = coco.getImgIds()

    # with open('image_ids_5k.json', 'r') as file:
    #     image_ids_5k = [int(id_str) for id_str in json.load(file)]
    
    # sampled_img_ids = random.sample(image_ids_5k, 20)

    # with open('./mscoco_val/annotations/captions_val2014.json') as f:
    #     captions_data = json.load(f)
    
    # # for img_info, annotation in zip(captions_data['images'], captions_data['annotations']):

    # model_image = timm.create_model('resnet50', pretrained=True)
    # model_image.eval()
    # model_text = DistilBertModel.from_pretrained('distilbert-base-uncased')
    # tokenizer_text = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    # model_text.eval()

    # transform = transforms.Compose([
    #     transforms.Resize(256),
    #     transforms.CenterCrop(224),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    # ])

    # image_embeddings = []
    # text_embeddings = []
    # for img_info, annotation in zip(captions_data['images'], captions_data['annotations']):
    #     if img_info['id'] not in sampled_img_ids:
    #         continue

    #     img_path = os.path.join(images_path, img_info['file_name'])
    #     image = Image.open(img_path).convert('RGB')
    #     image = transform(image).unsqueeze(0)

    #     with torch.no_grad():
    #         image_embeddings.append(model_image(image))
        
    #     caption = annotation['caption']
    #     inputs = tokenizer_text(caption, return_tensors='pt')

    #     with torch.no_grad():
    #         text_embeddings.append(model_text(**inputs).last_hidden_state.mean(dim=1))
    
    # image_feats = np.array(image_embeddings)
    # text_feats = np.array(text_embeddings)

    # image_centroids, text_centroids = kmeans_centroids(image_feats, text_feats, 5)

    

    # # For plotting the images

    # # print(sampled_img_ids)

    # # plt.figure(figsize=(20, 10))

    # # for i, image_id in enumerate(sampled_img_ids):
    # #     image_info = coco.loadImgs(image_id)[0]
    # #     image_path = os.path.join(dataDir, dataType, image_info['file_name'])
    # #     img = plt.imread(image_path)

    # #     ax = plt.subplot(4, 5, i + 1)
    # #     plt.imshow(img)
    # #     ax.axis('off')

    # # plt.show()





