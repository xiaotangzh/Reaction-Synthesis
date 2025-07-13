import numpy as np
import torch
from scipy.linalg import sqrtm

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns

def FID(realInput, realOutput, predOutput, network):  # use output encoder to calculate
    """
    Input: (N, latent_dim)
    Return: FID
    """
    
    real = network(realInput, t=realOutput, knn=torch.zeros(1).cuda())[1]
    pred = network(realInput, t=predOutput, knn=torch.zeros(1).cuda())[1]
    
    real = real.detach().cpu().numpy()
    pred = pred.detach().cpu().numpy()

    mu_real = np.mean(real, axis=0)
    sigma_real = np.cov(real, rowvar=False)
    # print(mu_real, sigma_real)

    mu_gen = np.mean(pred, axis=0)
    sigma_gen = np.cov(pred, rowvar=False)
    # print(mu_gen, sigma_gen)
    
    diff = mu_real - mu_gen
    diff_squared = np.sum(diff ** 2)

    covmean, _ = sqrtm(sigma_real.dot(sigma_gen), disp=False)

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid_value = diff_squared + np.trace(sigma_real + sigma_gen - 2 * covmean)
    print("FID: ", fid_value)
    return fid_value

def MatchingError(real, pred):
    f = torch.nn.MSELoss()
    error = f(real, pred)
    print("Matching error: ", error)
    return error


def DrawtSNE(target, estimate):
    if isinstance(target, torch.Tensor): target = target.detach().cpu().numpy()
    if isinstance(estimate, torch.Tensor): estimate = estimate.detach().cpu().numpy()
    
    n_samples1 = target.shape[0]
    n_samples2 = estimate.shape[0]
    
    target_labels = np.zeros(n_samples1)
    estimate_labels = np.ones(n_samples2)

    features = np.vstack((target, estimate))
    labels = np.hstack((target_labels, estimate_labels))

    tsne = TSNE(n_components=2, perplexity=30, random_state=42, n_iter=1000)
    features_2d = tsne.fit_transform(features)

    target_2d = features_2d[labels == 0]
    estimate_2d = features_2d[labels == 1]

    blue, red = (0.15, 0.5, 0.73, 0.5), (0.73, 0.24, 0.32, 0.5)
    plt.scatter(target_2d[:, 0], target_2d[:, 1], s=5, color=blue, label='Target Latent Space')
    plt.scatter(estimate_2d[:, 0], estimate_2d[:, 1], s=5, color=red, label='Estimated Latent Space')

    plt.title('t-SNE Visualization of Two Feature Sets')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.legend()
    plt.show()

def DrawtSNE_for3(target, estimate1, estimate2):
    if isinstance(target, torch.Tensor): target = target.detach().cpu().numpy()
    if isinstance(estimate1, torch.Tensor): estimate1 = estimate1.detach().cpu().numpy()
    if isinstance(estimate2, torch.Tensor): estimate2 = estimate2.detach().cpu().numpy()
    
    n_samples1 = target.shape[0]
    n_samples2 = estimate1.shape[0]
    n_samples3 = estimate2.shape[0]
    
    target_labels = np.zeros(n_samples1)
    estimate1_labels = np.ones(n_samples2)
    estimate2_labels = np.ones(n_samples3).fill(2)

    features = np.vstack((target, estimate1, estimate2))
    labels = np.hstack((target_labels, estimate1_labels, estimate2_labels))

    tsne = TSNE(n_components=2, perplexity=30, random_state=42, n_iter=1000)
    features_2d = tsne.fit_transform(features)

    target_2d = features_2d[labels == 0]
    estimate1_2d = features_2d[labels == 1]
    estimate2_2d = features_2d[labels == 2]

    blue, red = (0.15, 0.5, 0.73, 0.5), (0.73, 0.24, 0.32, 0.5)
    plt.scatter(target_2d[:, 0], target_2d[:, 1], s=5, color=blue, label='Target Latent Space')
    plt.scatter(estimate1_2d[:, 0], estimate1_2d[:, 1], s=5, color=red, label='Estimated Latent Space')
    plt.scatter(estimate2_2d[:, 0], estimate2_2d[:, 1], s=5, color=red, label='Estimated Latent Space')

    plt.title('t-SNE Visualization of Three Feature Sets')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.legend()
    plt.show()

def MPJPE(real, pred):
    # mpjpe = torch.norm(real[:,-468:]-pred[:,-468:])
    mpjpe = torch.mean(torch.norm(real-pred, dim=0))
    print("MPJPE: ", mpjpe)
    return mpjpe

def Diversity(realInput, realOutput, predOutput, network):
    real = network(realInput, t=realOutput, knn=torch.zeros(1).cuda())[1]
    pred = network(realInput, t=predOutput, knn=torch.zeros(1).cuda())[1]
    
    div = torch.mean(torch.var(pred, dim=0))
    print("Diversity: ", div)
    real_div = torch.mean(torch.var(real, dim=0))
    print("Ground Truth Diversity: ", real_div)
    return div

def ErrorOnIntention(real, pred):
    real, pred = real[:,:180], pred[:,:180]
    f = torch.nn.MSELoss()
    error = f(real, pred)
    print("Intention Error: ", error)
    return error

def DrawCodebookMatchingHeatmap(real, pred): # input: [N, 2048]
    real, pred = real.reshape(-1, 128, 16), pred.reshape(-1, 128, 16)
    error = torch.mean(torch.abs(real-pred), dim=0).detach().cpu().numpy()
    
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.figure(figsize=(4, 8))
    sns.heatmap(error, cmap='YlOrRd', cbar=True)
    plt.ylim(0, 0.35)
    plt.yticks(ticks=np.linspace(0, 128, 8), labels=np.round(np.linspace(0, 128, 8)).astype(int))
    plt.xticks(ticks=np.linspace(0, 16, 4), labels=np.round(np.linspace(0, 16, 4)).astype(int))
    plt.xlabel('Codebook Channels')
    plt.ylabel('Codebook Dimensions')
    plt.show()

def DrawMatchingErrorHistplot(array1, array2, array3):
    data = [array1.flatten(), array2.flatten(), array3.flatten()]
    labels = ['Ours', 'w/o GAN', 'Motion Matching']

    sns.set(style="whitegrid")

    plt.figure(figsize=(10, 6))
    colors = ['green', 'orange', 'blue']
    for i, d in enumerate(data):
            sns.histplot(d, kde=True, label=labels[i], bins=30, alpha=0.6, color=colors[i])

    plt.title('Codebook Matching Error with Users Control')
    plt.xlabel('Codebook Matching Error')
    plt.ylabel('Density')
    plt.legend() 
    plt.show()
    
def DrawTrajectoriesDistributionDifference(dist1=None, dist2=None):
    if dist1 is None and dist2 is None:
        dist1 = np.random.normal(loc=0, scale=1, size=(1000, 2))
        dist2 = np.random
        
    mean1, std1 = dist1[:, 0], dist1[:, 1]
    mean2, std2 = dist2[:, 0], dist2[:, 1]

    sns.set(style="whitegrid")

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    sns.histplot(mean1, kde=True, color='green', label='Joints trajectories in self root space', alpha=0.6, bins=30)
    sns.histplot(mean2, kde=True, color='red', label='Joints trajectories in relative root space', alpha=0.6, bins=30)
    plt.title('Mean Distribution Comparison')
    plt.xlabel('Mean')
    plt.ylabel('Density')
    plt.legend()

    plt.subplot(1, 2, 2)
    sns.histplot(std1, kde=True, color='green', label='Joints trajectories in self root space', alpha=0.6, bins=30)
    sns.histplot(std2, kde=True, color='red', label='Joints trajectories in relative root space', alpha=0.6, bins=30)
    plt.title('Standard Deviation Distribution Comparison')
    plt.xlabel('Standard Deviation')
    plt.ylabel('Density')
    plt.legend()

    plt.tight_layout()
    plt.show()  

def DrawMatchingError():
    x_values = [0, 100, 200, 300, 400, 500, 700, 1000, 1500, 2000]
    y_values = [0.054, 0.053, 0.051, 0.050, 0.049, 0.048, 0.052, 0.065, 0.090, 0.190]

    plt.figure(figsize=(8, 6))
    plt.plot(x_values, y_values, marker='o', linestyle='-', color='b')

    plt.yscale('log', base=2)
    
    y_ticks = [0.03, 0.05, 0.1, 0.2] 
    plt.yticks(y_ticks, labels=[f'{tick}' for tick in y_ticks])

    plt.xlabel('Ratio of GAN Loss to Matching Loss', fontsize=14)
    plt.ylabel('Codebook Matching Error (log base 2)', fontsize=14)

    plt.grid(True)
    plt.show()
    
def DrawLatentSpace():

    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    n_points = 100  
    t = np.linspace(0, 4 * np.pi, n_points)  
    x = np.sin(t) 
    y = np.cos(t)  
    z = t / (4 * np.pi) 

    noise_strength = 0.06  
    x += np.random.normal(0, noise_strength, n_points)
    y += np.random.normal(0, noise_strength, n_points)
    z += np.random.normal(0, noise_strength, n_points)

    colors = z 

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    sc = ax.scatter(x, y, z, c=colors, cmap='viridis', s=50)  

    ax.plot(x, y, z, color='gray', linewidth=1)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.grid(False)
    ax.set_axis_off()

    plt.colorbar(sc)
    plt.show()
