import numpy as np


X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])

y = np.array([[0],
              [1],
              [1],
              [0]])


np.random.seed(1)
w1 = np.random.randn(2, 4)   
b1 = np.zeros((1, 4))

w2 = np.random.randn(4, 1)  
b2 = np.zeros((1, 1))


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_turev(x):
    return x * (1 - x)


lr = 0.5


for i in range(1000):
    
    z1 = np.dot(X, w1) + b1
    a1 = sigmoid(z1)

    z2 = np.dot(a1, w2) + b2
    a2 = sigmoid(z2)

   
    hata = y - a2
    cikis_delta = hata * sigmoid_turev(a2)

    gizli_hata = np.dot(cikis_delta, w2.T)
    gizli_delta = gizli_hata * sigmoid_turev(a1)

   
    w2 += np.dot(a1.T, cikis_delta) * lr
    b2 += np.sum(cikis_delta, axis=0, keepdims=True) * lr

    w1 += np.dot(X.T, gizli_delta) * lr
    b1 += np.sum(gizli_delta, axis=0, keepdims=True) * lr


print("Tahminler:")
print(np.round(a2, 3))

print("\nEğitim sonrası ağırlıklar:")
print("w1 (giriş -> gizli):\n", w1)
print("w2 (gizli -> çıkış):\n", w2)


dogruluk = 1 - np.mean(np.abs(hata))
print("\nDoğruluk oranı: %", round(dogruluk * 100, 2))
