import numpy as np
import matplotlib.pyplot as plt

class Gaussian2DMap:
    def __init__(self, mus, covs, alphas):
        """
        mus: (N, 2) Gaussian centers
        covs: (N, 2, 2) covariance matrices
        alphas: (N,) opacity/weight scalars
        """
        self.mus = mus
        self.covs = covs
        self.alphas = alphas
        self.inv_covs = np.linalg.inv(covs)  # (N, 2, 2)

    def density(self, x):
        """
        x: (2,) or (M, 2) query point(s)
        return: density scalar or (M,) array
        """
        if x.ndim == 1:
            x = x[None, :]  # (1,2)
        diff = x[:, None, :] - self.mus[None, :, :]  # (M, N, 2)
        exp_terms = np.einsum('mni,nij,mnj->mn', diff, self.inv_covs, diff)  # (M, N)
        densities = self.alphas * np.exp(-0.5 * exp_terms)  # (M, N)
        return np.sum(densities, axis=1)  # (M,)
    
    def grad_density(self, x):  # x: (N, 2)
        diff = x[:, None, :] - self.mus[None, :, :]
        exp_terms = np.einsum('nmi,mij,nmj->nm', diff, self.inv_covs, diff)
        exp_vals = np.exp(-0.5 * exp_terms)
        weighted = self.alphas * exp_vals  # (N, M)
        grads = -np.einsum('nm,mij,nmj->ni', weighted, self.inv_covs, diff)  # (N, 2)
        return grads
    
    def density_with_cov(self, x):
        """
        x: (N,2)
        return:
            total_density: shape (N,)
            gradients: shape (N,2)
            mahalanobis_weighted: shape (N,)
        """
        N = x.shape[0]
        total_density = np.zeros(N)
        grad = np.zeros((N, 2))
        weighted_risk = np.zeros(N)

        for i in range(len(self.mus)):
            mu = self.mus[i]
            cov = self.covs[i]
            alpha = self.alphas[i]

            diff = x - mu
            cov_inv = np.linalg.inv(cov)
            d = np.einsum('ni,ij,nj->n', diff, cov_inv, diff)
            e = np.exp(-0.5 * d)

            total_density += alpha * e
            grad += alpha * e[:, None] * (-cov_inv @ diff.T).T  # shape (N,2)
            weighted_risk += alpha * e * d

        return total_density, grad, weighted_risk
    
    def mahalanobis_logbarrier(self, x, delta=1.0, max_grad=10.0):
        N = x.shape[0]
        cost = np.zeros(N)
        grad = np.zeros((N, 2))

        for i in range(len(self.mus)):
            mu = self.mus[i]
            cov = self.covs[i]
            alpha = self.alphas[i]

            cov_inv = np.linalg.inv(cov)
            diff = x - mu
            D = np.einsum('ni,ij,nj->n', diff, cov_inv, diff)
            eps = 1e-5
            d_safe = np.maximum(D - delta, eps)

            # Clipped gradient
            g = 1.0 / d_safe
            g = np.minimum(g, max_grad)

            cost += -alpha * np.log(d_safe)
            grad += -alpha * (2 * (cov_inv @ diff.T).T) * g[:, None]

        return cost, grad


if __name__ == "__main__":
    # 예시 Gaussian 데이터 (3개)
    mus = np.array([
        [2.0, 2.5],   # 장애물 cluster 1
        [2.5, 3.0],
        [3.0, 3.5],
        [4.0, 5.0],   # 장애물 cluster 2
        [5.0, 5.0],
        [6.0, 5.0],
        [5.0, 2.5],   # 장벽 형태
        [5.0, 3.0],
        [5.0, 3.5],
        [7.5, 6.0],   # 끝단 장애물
        [8.0, 6.5]
    ])

    # 방향성과 모양을 더 복잡하게 만든 공분산 행렬들
    covs = np.array([
        [[0.4, 0.2], [0.2, 0.3]],
        [[0.3, 0.1], [0.1, 0.2]],
        [[0.5, -0.1], [-0.1, 0.4]],
        [[0.2, 0.1], [0.1, 0.5]],
        [[0.6, -0.3], [-0.3, 0.6]],
        [[0.3, 0.0], [0.0, 0.3]],
        [[1.0, 0.0], [0.0, 0.1]],  # 긴 장벽
        [[1.0, 0.0], [0.0, 0.1]],
        [[1.0, 0.0], [0.0, 0.1]],
        [[0.3, 0.2], [0.2, 0.3]],
        [[0.4, -0.2], [-0.2, 0.4]]
    ])

    # 각각에 대한 중요도 (높은 값은 더 강한 장애물)
    alphas = np.array([
        0.8, 0.6, 0.7,
        0.9, 1.2, 0.9,
        1.0, 1.0, 1.0,
        0.7, 0.7
    ])

    # Gaussian map 생성
    gs_map = Gaussian2DMap(mus, covs, alphas)

    # 좌표 그리드 생성
    x = np.linspace(0, 10, 100)
    y = np.linspace(0, 10, 100)
    xx, yy = np.meshgrid(x, y)
    points = np.stack([xx.ravel(), yy.ravel()], axis=1)

    # 밀도 계산
    densities = gs_map.density(points)
    density_map = densities.reshape(xx.shape)

    # 결과 시각화
    plt.figure(figsize=(6,6))
    plt.contourf(xx, yy, density_map, levels=50, cmap='viridis')
    plt.colorbar(label='Density')
    plt.scatter(mus[:,0], mus[:,1], c='red', marker='x', label='Gaussian Centers')
    plt.title('2D Gaussian Splatting Map Density')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.show()
