import numpy as np


def obtain_y_hat_mix_ch(
        dataset, 
        W_,
        instance_norm = False, 
        bias = True
    ):
    L = len(dataset)
    X_ = []
    y_ = []
    for i in range(L):
        batch = dataset[i]
        x, y = batch[0], batch[1]
        X_.append(x)
        y_.append(y)

    X_ = np.array(X_)
    y_ = np.array(y_)

    X_train_mean = np.mean(X_, axis=1, keepdims=True)
    B, T_in, C = X_.shape
    B, T_out, C = y_.shape

    if instance_norm:
        X_ = X_ - X_train_mean
        y_ = y_ - X_train_mean

    if bias:
        IDim = T_in + 1
        X_ = np.concatenate((X_, np.ones((X_.shape[0], 1, X_.shape[-1]))), axis=1)
    else:
        IDim = T_in
        
    X_CI = X_.transpose(0,2,1).reshape((-1, IDim))
    y_CI = y_.transpose(0,2,1).reshape((-1, T_out))

    # compute prediction on train_set
    y_hat = (X_CI @ W_)

    return y_hat



def obtain_y_hat_indp_ch(
        dataset, 
        W_,
        instance_norm = False, 
        bias = True
    ):
    L = len(dataset)
    X_ = []
    y_ = []
    for i in range(L):
        batch = dataset[i]
        x, y = batch[0], batch[1]
        X_.append(x)
        y_.append(y)

    X_ = np.array(X_)
    y_ = np.array(y_)

    X_train_mean = np.mean(X_, axis=1, keepdims=True)
    B, T_in, C = X_.shape
    B, T_out, C = y_.shape

    if instance_norm:
        X_ = X_ - X_train_mean
        y_ = y_ - X_train_mean

    if bias:
        IDim = T_in + 1
        X_ = np.concatenate((X_, np.ones((X_.shape[0], 1, X_.shape[-1]))), axis=1)
    else:
        IDim = T_in
        
    X_CI = X_.transpose(0,2,1).reshape((-1, IDim))
    y_CI = y_.transpose(0,2,1).reshape((-1, T_out))

    # compute prediction on train_set
    # compute test results
    y_hat = np.matmul(X_.transpose(2, 0, 1).copy(), W_.transpose(2, 0, 1).copy()).transpose(1, 2, 0)
    
    return y_hat


def svd_mix_ch_linear(
        train_dataset, 
        test_dataset, 
        instance_norm = False, 
        bias = True,
        x_rank = None,
        y_rank = None,
        tol = 1e-9
    ):
    L = len(train_dataset)
    X_train = []
    y_train = []
    for i in range(L):
        batch = train_dataset[i]
        x_, y_ = batch[0], batch[1]
        X_train.append(x_)
        y_train.append(y_)

    X_train = np.array(X_train)
    y_train = np.array(y_train)

    X_train_mean = np.mean(X_train, axis=1, keepdims=True)
    B, T_in, C = X_train.shape
    B, T_out, C = y_train.shape

    if instance_norm:
        X_train = X_train - X_train_mean
        y_train = y_train - X_train_mean

    if bias:
        IDim = T_in + 1
        X_train = np.concatenate((X_train, np.ones((X_train.shape[0], 1, X_train.shape[-1]))), axis=1)
    else:
        IDim = T_in
        
    X_train_CI = X_train.transpose(0,2,1).reshape((-1, IDim))
    y_train_CI = y_train.transpose(0,2,1).reshape((-1, T_out))


    U, s, Vt = np.linalg.svd(X_train_CI, full_matrices=False)
    if x_rank is None:
        s_inv = np.array([1/si if si > tol else 0 for si in s])
    else:
        s[x_rank:] = 0
        s_inv = np.array([1/si if si > tol else 0 for si in s])

    if y_rank is None:
        Y = y_train_CI
    else:
        uy, sy, vty = np.linalg.svd(y_train_CI, full_matrices=False)
        sy[y_rank:] = 0
        Y = uy @ np.diag(sy) @ vty
    
    W = Vt.T @ np.diag(s_inv) @ U.T @ Y

    # create test_dataset
    L = len(test_dataset)
    X_test = []
    y_test = []
    for i in range(L):
        batch = test_dataset[i]
        x_, y_ = batch[0], batch[1]
        X_test.append(x_)
        y_test.append(y_)

    X_test = np.array(X_test)
    y_test = np.array(y_test)
    X_test_mean = np.mean(X_test, axis=1, keepdims=True)
    if instance_norm:
        X_test -= X_test_mean
    if bias:
        X_test = np.concatenate((X_test, np.ones((X_test.shape[0], 1, X_test.shape[-1]))), axis=1)

    # compute prediction on test_set
    X_test_CI = X_test.transpose(0,2,1).reshape((-1, IDim))
    y_hat_test = (X_test_CI @ W).reshape(-1, C, T_out).transpose(0,2,1) # BTC
    if instance_norm:
        y_hat_test = (X_test_CI @ W).reshape(-1, C, T_out).transpose(0,2,1) + X_test_mean 

    # find test mse
    mse_test = ((y_hat_test-y_test)**2).mean()
    mae_test = np.abs(y_hat_test - y_test).mean()

    # compute prediction on train_set
    y_hat_train = (X_train_CI @ W).reshape(-1, C, T_out).transpose(0,2,1)
    mse_train = ((y_hat_train - y_train)**2).mean()
    mae_train = np.abs(y_hat_train - y_train).mean()

    if instance_norm:
        result = (
            W,
            (y_hat_test, y_test, y_hat_train+X_train_mean, y_train+X_train_mean),
            (mse_test, mae_test, mse_train, mae_train),
        )
    else:
        result = (
            W,
            (y_hat_test, y_test, y_hat_train, y_train),
            (mse_test, mae_test, mse_train, mae_train),
        )

    return result






def svd_indp_ch_linear(
        train_dataset, 
        test_dataset, 
        instance_norm = False, 
        bias = True,
        x_rank = None,
        y_rank = None,
        tol = 1e-9
    ):
    L = len(train_dataset)
    X_train = []
    y_train = []
    for i in range(L):
        batch = train_dataset[i]
        x_, y_ = batch[0], batch[1]
        X_train.append(x_)
        y_train.append(y_)


    X_train = np.array(X_train)
    y_train = np.array(y_train)
    B, T_in, C = X_train.shape 
    B, T_out, C = y_train.shape 
    X_train_mean = np.mean(X_train, axis=1, keepdims=True)
    if instance_norm:
        X_train = X_train - X_train_mean
        y_train -= X_train_mean

    if bias:
        X_train = np.concatenate((X_train, np.ones((X_train.shape[0], 1, X_train.shape[-1]))), axis=1)
        in_dim = T_in+1
    else:
        in_dim = T_in
        
    W = []
    C = X_train.shape[-1]

    for i in range(C):
        U, s, Vt = np.linalg.svd(X_train[...,i].copy(), full_matrices=False)
        if x_rank is None:
            s_inv = np.array([1/si if si > tol else 0 for si in s])
        else:
            s[x_rank:] = 0
            s_inv = np.array([1/si if si > tol else 0 for si in s])

        if y_rank is None:
            Y = y_train[...,i].copy()
        else:
            uy, sy, vty = np.linalg.svd(y_train[...,i].copy(), full_matrices=False)
            sy[y_rank:] = 0
            Y = uy @ np.diag(sy) @ vty
        
        W_ = Vt.T @ np.diag(s_inv) @ U.T @ Y

        W.append(W_)
    W = np.array(W)
    W = W.transpose(1,2,0)


    # gather test dataset
    L = len(test_dataset)
    X_test = []
    y_test = []
    for i in range(L):
        batch = test_dataset[i]
        x_, y_ = batch[0], batch[1]
        X_test.append(x_)
        y_test.append(y_)

    X_test = np.array(X_test)
    y_test = np.array(y_test)
    X_test_mean = np.mean(X_test, axis=1, keepdims=True)
    if instance_norm:
        X_test -= X_test_mean
    if bias:
        X_test = np.concatenate((X_test, np.ones((X_test.shape[0], 1, X_test.shape[-1]))), axis=1)

    # compute test results
    y_hat_test = np.einsum('btc,toc->boc', X_test, W)
    if instance_norm:
        y_hat_test = np.einsum('btc,toc->boc', X_test, W) + X_test_mean

    mse_test = ((y_hat_test-y_test)**2).mean()
    mae_test = np.abs(y_hat_test - y_test).mean()

    # compute train results
    y_hat_train = np.einsum('btc,toc->boc', X_train, W)
    mse_train = ((y_hat_train - y_train)**2).mean()
    mae_train = np.abs(y_hat_train - y_train).mean()

    # gather results
    if instance_norm:
        result = (
            W,
            (y_hat_test, y_test, y_hat_train+X_train_mean, y_train+X_train_mean),
            (mse_test, mae_test, mse_train, mae_train),
        )
    else:
        result = (
            W,
            (y_hat_test, y_test, y_hat_train, y_train),
            (mse_test, mae_test, mse_train, mae_train),
        )

    return result




def ols_mix_ch_linear(
        train_dataset, 
        test_dataset, 
        instance_norm = False, 
        mean_factor = 1.0, 
        lambda_ = 0.,
        bias = True
    ):
    L = len(train_dataset)
    X_train = []
    y_train = []
    for i in range(L):
        batch = train_dataset[i]
        x_, y_ = batch[0], batch[1]
        X_train.append(x_)
        y_train.append(y_)

    X_train = np.array(X_train)
    y_train = np.array(y_train)

    X_train_mean = np.mean(X_train, axis=1, keepdims=True) * mean_factor
    B, T_in, C = X_train.shape
    B, T_out, C = y_train.shape

    if instance_norm:
        X_train = X_train - X_train_mean + np.random.randn(*X_train.shape)*0.01
        y_train -= X_train_mean

    if bias:
        IDim = T_in + 1
        X_train = np.concatenate((X_train, np.ones((X_train.shape[0], 1, X_train.shape[-1]))), axis=1)
    else:
        IDim = T_in
        
    X_train_CI = X_train.transpose(0,2,1).reshape((-1, IDim))
    y_train_CI = y_train.transpose(0,2,1).reshape((-1, T_out))
    W = np.linalg.inv(X_train_CI.T @ X_train_CI + lambda_ * np.eye(IDim)) @ X_train_CI.T @ y_train_CI
    eigenvalues, eigenvectors = np.linalg.eig(X_train_CI.T @ X_train_CI)

    # create test_dataset
    L = len(test_dataset)
    X_test = []
    y_test = []
    for i in range(L):
        batch = test_dataset[i]
        x_, y_ = batch[0], batch[1]
        X_test.append(x_)
        y_test.append(y_)

    X_test = np.array(X_test)
    y_test = np.array(y_test)
    X_test_mean = np.mean(X_test, axis=1, keepdims=True) * mean_factor
    if instance_norm:
        X_test -= X_test_mean
    if bias:
        X_test = np.concatenate((X_test, np.ones((X_test.shape[0], 1, X_test.shape[-1]))), axis=1)

    # compute prediction on test_set
    X_test_CI = X_test.transpose(0,2,1).reshape((-1, IDim))
    y_hat_test = (X_test_CI @ W).reshape(-1, C, T_out).transpose(0,2,1) # BTC
    if instance_norm:
        y_hat_test = (X_test_CI @ W).reshape(-1, C, T_out).transpose(0,2,1) + X_test_mean 

    # find test mse
    mse_test = ((y_hat_test-y_test)**2).mean()
    mae_test = np.abs(y_hat_test - y_test).mean()

    # compute prediction on train_set
    y_hat_train = (X_train_CI @ W).reshape(-1, C, T_out).transpose(0,2,1)
    mse_train = ((y_hat_train - y_train)**2).mean()
    mae_train = np.abs(y_hat_train - y_train).mean()

    if instance_norm:
        result = (
            W,
            (y_hat_test, y_test, y_hat_train+X_train_mean, y_train+X_train_mean),
            (mse_test, mae_test, mse_train, mae_train),
            (eigenvalues, eigenvectors)
        )
    else:
        result = (
            W,
            (y_hat_test, y_test, y_hat_train, y_train),
            (mse_test, mae_test, mse_train, mae_train),
            (eigenvalues, eigenvectors)
        )

    return result


def infer_ols_mixed_ch(
        test_dataset,
        W_,
        instance_norm = False,
        bias = False
    ):

    # create test_dataset
    L = len(test_dataset)
    X_test = []
    y_test = []
    for i in range(L):
        batch = test_dataset[i]
        x_, y_ = batch[0], batch[1]
        X_test.append(x_)
        y_test.append(y_)

    X_test = np.array(X_test)
    y_test = np.array(y_test)

    B, T_in, C = X_test.shape
    B, T_out, C = y_test.shape

    X_test_mean = np.mean(X_test, axis=1, keepdims=True)
    if instance_norm:
        X_test -= X_test_mean

    if bias:
        X_test = np.concatenate((X_test, np.ones((X_test.shape[0], 1, X_test.shape[-1]))), axis=1)
        IDim = T_in+1
    else:
        IDim = T_in

    # compute prediction on test_set
    X_test_CI = X_test.transpose(0,2,1).reshape((-1, IDim))
    if instance_norm:
        y_hat_test = (X_test_CI @ W_).reshape(-1, C, T_out).transpose(0,2,1) + X_test_mean 
    else:
        y_hat_test = (X_test_CI @ W_).reshape(-1, C, T_out).transpose(0,2,1) # BTC

    # find test mse
    mse_test = ((y_hat_test-y_test)**2).mean()
    mae_test = np.abs(y_hat_test - y_test).mean()

    return ((mse_test, mae_test), (y_hat_test, y_test))



def ols_indp_ch_linear(
        train_dataset, 
        test_dataset, 
        instance_norm = False, 
        mean_factor = 1.0, 
        lambda_ = 0.,
        bias = True
    ):
    L = len(train_dataset)
    X_train = []
    y_train = []
    for i in range(L):
        batch = train_dataset[i]
        x_, y_ = batch[0], batch[1]
        X_train.append(x_)
        y_train.append(y_)


    X_train = np.array(X_train)
    y_train = np.array(y_train)
    B, T_in, C = X_train.shape 
    B, T_out, C = y_train.shape 
    X_train_mean = np.mean(X_train, axis=1, keepdims=True)*mean_factor
    if instance_norm:
        X_train = X_train - X_train_mean + np.random.randn(*X_train.shape)*0.01
        y_train -= X_train_mean

    if bias:
        X_train = np.concatenate((X_train, np.ones((X_train.shape[0], 1, X_train.shape[-1]))), axis=1)
        in_dim = T_in+1
    else:
        in_dim = T_in
        
    W = []
    C = X_train.shape[-1]
    for i in range(C):
        W.append(
            np.linalg.inv(X_train[...,i].T.copy() @ X_train[...,i].copy() + lambda_*np.eye(in_dim)) @ \
                X_train[...,i].T.copy() @ y_train[...,i].copy()
        )
    W = np.array(W)
    W = W.transpose(1,2,0)

    eigenvalues, eigenvectors = [], []
    for i in range(C):
        eigenvalues_, eigenvectors_ = np.linalg.eig(X_train[...,i].T.copy() @ X_train[...,i].copy())
        eigenvalues.append(eigenvalues_)
        eigenvectors.append(eigenvectors_)

    # gather test dataset
    L = len(test_dataset)
    X_test = []
    y_test = []
    for i in range(L):
        batch = test_dataset[i]
        x_, y_ = batch[0], batch[1]
        X_test.append(x_)
        y_test.append(y_)

    X_test = np.array(X_test)
    y_test = np.array(y_test)
    X_test_mean = np.mean(X_test, axis=1, keepdims=True)*mean_factor
    if instance_norm:
        X_test -= X_test_mean
    if bias:
        X_test = np.concatenate((X_test, np.ones((X_test.shape[0], 1, X_test.shape[-1]))), axis=1)

    # compute test results
    y_hat_test = np.einsum('btc,toc->boc', X_test, W)
    if instance_norm:
        y_hat_test = np.einsum('btc,toc->boc', X_test, W) + X_test_mean

    mse_test = ((y_hat_test-y_test)**2).mean()
    mae_test = np.abs(y_hat_test - y_test).mean()

    # compute train results
    y_hat_train = np.einsum('btc,toc->boc', X_train, W)
    mse_train = ((y_hat_train - y_train)**2).mean()
    mae_train = np.abs(y_hat_train - y_train).mean()

    # gather results
    if instance_norm:
        result = (
            W,
            (y_hat_test, y_test, y_hat_train+X_train_mean, y_train+X_train_mean),
            (mse_test, mae_test, mse_train, mae_train),
            (eigenvalues, eigenvectors)
        )
    else:
        result = (
            W,
            (y_hat_test, y_test, y_hat_train, y_train),
            (mse_test, mae_test, mse_train, mae_train),
            (eigenvalues, eigenvectors)
        )

    return result


def infer_ols_indp_ch(
        test_dataset,
        W_,
        instance_norm = False,
        bias = False
    ):

    # create test_dataset
    L = len(test_dataset)
    X_test = []
    y_test = []
    for i in range(L):
        batch = test_dataset[i]
        x_, y_ = batch[0], batch[1]
        X_test.append(x_)
        y_test.append(y_)

    X_test = np.array(X_test)
    y_test = np.array(y_test)

    B, T_in, C = X_test.shape
    B, T_out, C = y_test.shape

    X_test_mean = np.mean(X_test, axis=1, keepdims=True)
    if instance_norm:
        X_test -= X_test_mean

    if bias:
        X_test = np.concatenate((X_test, np.ones((X_test.shape[0], 1, X_test.shape[-1]))), axis=1)
        IDim = T_in+1
    else:
        IDim = T_in

    # compute test results
    
    if instance_norm:
        y_hat_test = np.matmul(X_test.transpose(2, 0, 1).copy(), W_.transpose(2, 0, 1).copy()).transpose(1, 2, 0) + X_test_mean
    else:
        y_hat_test = np.matmul(X_test.transpose(2, 0, 1).copy(), W_.transpose(2, 0, 1).copy()).transpose(1, 2, 0)

    mse_test = ((y_hat_test-y_test)**2).mean()
    mae_test = np.abs(y_hat_test - y_test).mean()

    return ((mse_test, mae_test), (y_hat_test, y_test))