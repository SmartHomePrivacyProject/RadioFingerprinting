# Robust Deep-learning-based Radio Fingerprinting with Fine-Tuning

## Overview 

In this project, we study radio fingerprinting. Radio fingerprinting is a way of distinguishing a tramisster from others of the same type by analyzing raw I/Q data  in wireless communications. Each tramisster has a unique finerprint due to hardware imperfections on radio-frequency circuitry. The hardware imperhections include I/Q imbalance, phase noise, frequency offset, and sampling off-set. Specifically, given I/Q data collected on a receiver side, a receiver decides which transmitter it is in radio fingerprinting. 

Recent studies have shown that deep learning can obtain high accuracy in radio fingerpinting. However, as wireless channels keep changing in the real world, one primary challenge is that the deep nueral networks for radio fingerprinting are not robust, where the training I/Q data are different from test I/Q data, especially in a cross-day scenario. We leverage fine-tuning to improve the robustness of dee-learning-based radio fingerprinting. 

## Testbed
