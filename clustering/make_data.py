from scipy.stats import multivariate_normal, norm, expon
import numpy as np

N_SAMPLES = 500
N_FEATURES = 100
N_CLUSTERS = 5
PARAM = 0.5
TEST_SIZE = 0.2


def assign_clusters(n=N_SAMPLES, c=N_CLUSTERS, param=PARAM):
    availible_inds = np.array(range(n))
    n_remaining = n
    assignments = np.zeros(n, dtype=np.int32)
    mean_size = int(n/c)
    sd = param*mean_size

    for i in range(c):
        cl_size = min(int(norm.rvs(mean_size, sd, 1)), n_remaining)
        if i == (c-1) or cl_size > n_remaining:
            cl_size = n_remaining
        choices = np.random.choice(availible_inds, cl_size, replace=False)
        for ch in choices:
            assignments[ch] = i
        n_remaining = n_remaining - cl_size
        availible_inds = np.setdiff1d(availible_inds, choices)

    return assignments


# def create_data(n=N_SAMPLES, c=N_CLUSTERS, f=N_FEATURES):
#     features = np.zeros([n,f])
#     feat_means = np.array([expon.rvs(size=f) for _ in range(c)])
#
#     cluster_assignments = assign_clusters(n,c)
#     for i, cl_i in enumerate(cluster_assignments):
#         features[i,:] = expon.rvs(scale=feat_means[cl_i,:], size=f)





def create_data(n=N_SAMPLES, c=N_CLUSTERS, f=N_FEATURES, param=PARAM, time_str=""):
    feat_means = np.array([expon.rvs(size=f) for _ in range(c)])

    feat_clusters = assign_clusters(f, c, param)
    cov = np.zeros([f,f])
    for ii in range(f):
        for jj in range(ii,f):
            if feat_clusters[ii] == feat_clusters[jj]:
                cv = expon.rvs(size=1)
                cov[ii,jj] = cv
                cov[jj,ii] = cv

    # feat_sds = np.array([expon.rvs(size=f) for _ in range(c)])

    cluster_assignments = assign_clusters(n, c, param)
    cluster_assignments_test = assign_clusters(TEST_SIZE, c, param)

    features = create_features(feat_means, cov, cluster_assignments)
    features_test = create_features(feat_means, cov, cluster_assignments_test)

    labels = cluster_assignments%2
    labels_test = cluster_assignments_test%2

    fstr = "data/{}_" + time_str

    np.save(fstr.format("importances"), feat_means)
    np.save(fstr.format("X"), features)
    np.save(fstr.format("y"), labels)
    np.save(fstr.format("X_test"), features_test)
    np.save(fstr.format("y_test"), labels_test)
    np.save(fstr.format("cl"), cluster_assignments)
    np.save(fstr.format("cl_test"), cluster_assignments_test)


    return features, labels


def get_time():
    import time
    t = time.localtime()
    return "{}_{}".format(t.tm_min, t.tm_sec)


def create_features(means, sds, cluster_assignments=None):

    f = means.shape[1]


    # if not cluster_assignments.any() == None: # treat each point as own cluster
    #     cluster_assignments = np.array(list(range(f)))

    features = np.zeros([len(cluster_assignments),f])
    for i, cl_i in enumerate(cluster_assignments):
        new = multivariate_normal.rvs(mean=means[cl_i, :])

        features[i, :] = new
    return features



if __name__ == '__main__':
    if TEST_SIZE < 1:
        TEST_SIZE = int(TEST_SIZE*N_SAMPLES)

    time_str = get_time()
    training = create_data(time_str=time_str)

    print("TIME_STR:", time_str)


