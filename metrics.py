#accuracy utilities

import numpy as np
from typing import Tuple

def _dilate_1d_binary(x: np.ndarray, radius: int) -> np.ndarray:
    """
    Simple 1D binary dilation by a flat window of size (2*radius+1).
    """
    x = np.asarray(x).astype(np.uint8).reshape(-1)
    if radius <= 0:
        return x
    k = 2 * radius + 1
    pad = np.pad(x, (radius, radius), mode="edge")
    out = np.empty_like(x)
    for i in range(len(x)):
        out[i] = pad[i:i+k].max()
    return out

def binary_accuracy_with_neighborhood(
    preds: np.ndarray,
    targets: np.ndarray,
    threshold: float = 0.10,
    radius: int = 0
) -> Tuple[float, Tuple[int, int, int, int]]:
    """
    Paper Acc (%): threshold both signals at `threshold`, optionally dilate by `radius`,
    then Acc = (TP + TN) / (TP + TN + FP + FN) * 100.
    Returns (acc_percent, (TP, TN, FP, FN)).
    """
    p = np.asarray(preds).reshape(-1)
    t = np.asarray(targets).reshape(-1)
    p_bin = (p > threshold).astype(np.uint8)
    t_bin = (t > threshold).astype(np.uint8)

    if radius > 0:
        p_bin = _dilate_1d_binary(p_bin, radius)
        t_bin = _dilate_1d_binary(t_bin, radius)

    TP = int(((p_bin == 1) & (t_bin == 1)).sum())
    TN = int(((p_bin == 0) & (t_bin == 0)).sum())
    FP = int(((p_bin == 1) & (t_bin == 0)).sum())
    FN = int(((p_bin == 0) & (t_bin == 1)).sum())
    denom = TP + TN + FP + FN
    acc = 100.0 * (TP + TN) / denom if denom else 0.0
    return acc, (TP, TN, FP, FN)



def leaky_accuracy(y_pred, y_true, span=3, threshold=0.05):
    assert span > 1
    #assert epsilon >= 0.0
    #assert y_pred.shape.__len__() == 1
    #assert y_true.shape.__len__() == 1
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()

    #difference = np.abs(y_pred - y_true).__le__(epsilon).reshape(-1, 1)
    #  output = [joystick_erp[:joy_idx + 1].max(keepdims=True) for joy_idx in range(joystick_erp.shape[0])]
    #target = y_true.__ge__(epsilon).flatten()
    pred = y_pred > threshold
    true = y_true > threshold

    FP =  pred & ~true
    #TP =  pred &  true
    FN = ~pred &  true
    TN = ~pred & ~true

    rshift = [np.concatenate([np.zeros(i, dtype=bool), pred[:-i]]) for i in range(1, min(span, len(true)) + 1)]
    lshift = [np.concatenate([pred[i:], np.zeros(i, dtype=bool)]) for i in range(1, min(span, len(true)) + 1)]
    
    TP = (np.stack([*lshift, pred, *rshift]) & true.reshape((1, -1))).any(0)

    acc = (TP | TN).astype(float).mean()
    prec = TP.astype(float).mean() / (TP | FP).astype(float).mean()
    recl = TP.astype(float).mean() / (TP | FN).astype(float).mean()
    f1 = 2 * (prec * recl) / (prec + recl)

    return acc, prec, recl, f1


if __name__ == '__main__':
    output = [[0.0, 0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.4, 0.0]]
    target = [[ .0, 0, 0, 0, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4]]
    print('accuracy', leaky_accuracy(np.array(output), np.array(target)))


    output = [2.840776687662583e-05, 2.7496724214870483e-05, 2.8048794774804264e-05, 2.794088504742831e-05, 2.693711758183781e-05, 2.976002542709466e-05, 2.9570142942247912e-05, 2.818175380525645e-05, 2.7603786293184385e-05, 2.698144453461282e-05, 2.7740432415157557e-05, 2.82325654552551e-05, 3.1146588298724964e-05, 3.0597115255659446e-05, 2.865607530111447e-05, 3.0131674066069536e-05, 3.2570085750194266e-05, 3.056521381950006e-05, 3.083764386246912e-05, 2.7935528123634867e-05, 3.466392809059471e-05, 3.901164745911956e-05, 3.83157967007719e-05, 4.658328543882817e-05, 4.7162808186840266e-05, 3.4100114135071635e-05, 3.576645758585073e-05, 3.296512659289874e-05, 3.796697637881152e-05, 3.324683348182589e-05, 3.030839070561342e-05, 2.7797528673545457e-05, 2.7219777621212415e-05, 2.948915607703384e-05, 2.815532025124412e-05, 2.7741938538383693e-05, 2.7851576305693015e-05, 2.8814603865612298e-05, 3.3615891879890114e-05, 3.706240386236459e-05, 3.4644595871213824e-05, 2.7978478101431392e-05, 2.6613299269229174e-05, 2.6914496629615314e-05, 2.6453077225596644e-05, 2.6251980671077035e-05, 2.683341153897345e-05, 2.44370039581554e-05]
    target = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    output = (np.array(output) > 0.1).astype(np.float64)
    target = np.array(target)

    print('accuracy', leaky_accuracy(output, target))
